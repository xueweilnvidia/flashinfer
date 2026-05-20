#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sweep autotune knobs for ``flashinfer.norm.fused_rmsnorm_silu``.

For each unique ``(C, num_tokens)`` pair extracted from a shapes file, enumerate
the kernel knob space ``(warps_m, split_cols, kernel_cfg, occupancy, bpl)``,
JIT-compile every valid combination, refcheck against an fp32 PyTorch reference,
and benchmark with CUDA events. Prints the fastest config per shape in a form
ready to paste into ``_KNOB_LUT`` in ``flashinfer/jit/rmsnorm_silu.py``.

Notes
-----
- JIT compile is the dominant cost (~10-30 s per unique knob set). Compiles are
  ``@functools.cache``-d in ``flashinfer.norm`` and shared across shapes with
  the same ``C``, so total wall time scales roughly with the number of unique
  ``(C, knobs)`` pairs.
- A first-time full sweep on 9 shapes / 3 unique C values takes ~1-2 hours on
  a single GPU. Re-running is fast (compiles are cached on disk).
- ``--quick`` reduces the search space (~10x fewer candidates) for sanity
  checks.

Usage
-----
    # Full sweep over the dump file produced by dump_wan_vae_fused_shapes.py
    python sweep_fused_rmsnorm_silu_knobs.py \
        --shapes-file wan_vae_fused_rmsnorm_silu_shapes.txt \
        --csv sweep_results.csv

    # Quick sanity check
    python sweep_fused_rmsnorm_silu_knobs.py --quick

    # Single shape (C, num_tokens)
    python sweep_fused_rmsnorm_silu_knobs.py --shape 96,2129920
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
import traceback
from typing import List, Optional, Tuple

import torch

# Internal helpers — these are intentionally non-public but stable enough for
# a one-shot tuning script.
from flashinfer.norm import (
    _compute_rmsnorm_silu_workspace_size,
    _get_rmsnorm_silu_module,
    _get_rmsnorm_silu_sm_count,
)
from flashinfer.jit.rmsnorm_silu import _estimate_ctas_per_row


# ----------------------------------------------------------------------------
# Shape parsing
# ----------------------------------------------------------------------------


def parse_shape_file(path: str) -> List[Tuple[int, int]]:
    """Read (num_tokens, hidden_size) from the dump file, deduped, file order."""
    out: List[Tuple[int, int]] = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            num_tokens = int(parts[0])
            C = int(parts[1])
            key = (C, num_tokens)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


# ----------------------------------------------------------------------------
# Reference
# ----------------------------------------------------------------------------


def reference_rmsnorm_silu(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """fp32 RMSNorm + SiLU reference. Returns bf16."""
    x32 = x.float()
    rms = torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + eps)
    y = x32 * rms * weight.float()
    out = y * torch.sigmoid(y)
    return out.to(x.dtype)


# ----------------------------------------------------------------------------
# Candidate enumeration
# ----------------------------------------------------------------------------


# Observed in the existing LUT (flashinfer/jit/rmsnorm_silu.py).
_FULL_WARPS_M = (1, 4, 8, 32)
_FULL_OCCUPANCY = (1, 2, 3, 4, 5, 6, 8, 9, 10, 16)
_FULL_KERNEL_CFG = (0, 1, 2)
_FULL_BPL = (2, 4, 8, 16)
_FULL_SPLIT_COLS = (0, 4)

_QUICK_WARPS_M = (1, 8, 32)
_QUICK_OCCUPANCY = (1, 2, 4, 8)
_QUICK_KERNEL_CFG = (0, 1, 2)
_QUICK_BPL = (2, 4, 8, 16)
_QUICK_SPLIT_COLS = (0, 4)


def _valid_for_ctas_per_row_1(C: int, bpl: int) -> bool:
    """Mirror the validity check in `_compute_default_knobs` for ctas_per_row=1."""
    if bpl < 2 or bpl % 2 != 0:
        return False
    num_elts = bpl // 2  # bf16
    if num_elts <= 0 or C % num_elts != 0:
        return False
    vec_cols = C // num_elts
    vec_cols_per_ldg = 1 * 1 * 32  # ctas_per_row * warps_n * 32
    if vec_cols % vec_cols_per_ldg != 0:
        return False
    ldgs = vec_cols // vec_cols_per_ldg
    return 0 < ldgs <= 1024


def enumerate_candidates(C: int, quick: bool) -> List[Tuple[int, int, int, int, int]]:
    """Return list of (warps_m, split_cols, kernel_cfg, occupancy, bpl).

    Only combinations that pass `_estimate_ctas_per_row` and the kernel's
    vectorization constraint are included. We deduplicate by the effective
    `(warps_m, ctas_per_row, bpl, kernel_cfg, occupancy)` tuple because that
    is what `_get_rmsnorm_silu_module` actually keys on.
    """
    warps_m_vals = _QUICK_WARPS_M if quick else _FULL_WARPS_M
    occ_vals = _QUICK_OCCUPANCY if quick else _FULL_OCCUPANCY
    cfg_vals = _QUICK_KERNEL_CFG if quick else _FULL_KERNEL_CFG
    bpl_vals = _QUICK_BPL if quick else _FULL_BPL
    split_vals = _QUICK_SPLIT_COLS if quick else _FULL_SPLIT_COLS

    out: List[Tuple[int, int, int, int, int]] = []
    seen_module_key = set()
    for bpl, warps_m, split_cols, kernel_cfg, occupancy in itertools.product(
        bpl_vals, warps_m_vals, split_vals, cfg_vals, occ_vals
    ):
        ctas_per_row = _estimate_ctas_per_row(C, split_cols, kernel_cfg, bpl)
        # When ctas_per_row resolves to 1, split_cols=0 vs 4 produce identical
        # modules; skip the duplicate.
        if ctas_per_row == 1 and split_cols == 4:
            continue
        # Validate against the kernel's vector-load requirement.
        if ctas_per_row == 1:
            if not _valid_for_ctas_per_row_1(C, bpl):
                continue
        # The actual JIT cache key.
        mod_key = (warps_m, ctas_per_row, bpl, kernel_cfg, occupancy)
        if mod_key in seen_module_key:
            # Different split_cols collapsed to same module — skip duplicate.
            continue
        seen_module_key.add(mod_key)
        out.append((warps_m, split_cols, kernel_cfg, occupancy, bpl))
    return out


# ----------------------------------------------------------------------------
# Bench one candidate
# ----------------------------------------------------------------------------


def bench_one_candidate(
    C: int,
    num_tokens: int,
    knobs: Tuple[int, int, int, int, int],
    x: torch.Tensor,
    weight: torch.Tensor,
    ref_out: torch.Tensor,
    sm_count: int,
    warmup: int,
    iters: int,
    err_tol: float,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Compile + smoke-test + benchmark one knob set.

    Returns (mean_time_us, max_err, error_msg). On success, error_msg is None.
    """
    warps_m, split_cols, kernel_cfg, occupancy, bpl = knobs
    ctas_per_row = _estimate_ctas_per_row(C, split_cols, kernel_cfg, bpl)
    device = x.device

    try:
        module = _get_rmsnorm_silu_module(
            C, "bf16", warps_m, ctas_per_row, bpl, kernel_cfg, occupancy
        )
    except Exception as e:
        return None, None, f"compile_fail:{type(e).__name__}"

    try:
        ws_size = _compute_rmsnorm_silu_workspace_size(
            num_tokens,
            C,
            "bf16",
            warps_m,
            ctas_per_row,
            kernel_cfg,
            occupancy,
            sm_count,
        )
        workspace = torch.empty(ws_size, dtype=torch.uint8, device=device)
        out = torch.empty_like(x)
        scale_row_out = torch.empty(0, dtype=torch.uint8, device=device)

        # Smoke run + refcheck.
        module.rmsnorm_silu(out, x, weight, 1e-6, workspace, scale_row_out, sm_count)
        torch.cuda.synchronize()
        err = (out.float() - ref_out.float()).abs().max().item()
        if not (err < err_tol):  # also catches NaN
            return None, err, "refcheck_fail"

        # Warmup.
        for _ in range(warmup):
            module.rmsnorm_silu(
                out, x, weight, 1e-6, workspace, scale_row_out, sm_count
            )
        torch.cuda.synchronize()

        # Timed loop.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module.rmsnorm_silu(
                out, x, weight, 1e-6, workspace, scale_row_out, sm_count
            )
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) / iters * 1000.0  # ms → us
        return us, err, None
    except Exception as e:
        return None, None, f"run_fail:{type(e).__name__}"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--shapes-file", default="wan_vae_fused_rmsnorm_silu_shapes.txt"
    )
    p.add_argument(
        "--shape",
        help="Override shapes file with a single 'C,num_tokens' pair "
        "(e.g. --shape 96,2129920)",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--err-tol",
        type=float,
        default=0.2,
        help="Max-abs refcheck tolerance vs fp32 RMSNorm+SiLU (default 0.2). "
        "bf16 noise can reach ~1/16 ≈ 0.0625, so 0.2 leaves margin.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Reduce candidate set for fast sanity checks.",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="Write full sweep results (one row per candidate) to this CSV.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Print top-K configs per shape (default 5).",
    )
    p.add_argument(
        "--skip-c",
        default="",
        help="Comma-separated list of C (hidden_size) values to skip.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    sm_count = _get_rmsnorm_silu_sm_count(device.index)
    cc = torch.cuda.get_device_capability(device.index)
    print(
        f"# device : {torch.cuda.get_device_name(device.index)} "
        f"(sm{cc[0]}{cc[1]}, {sm_count} SMs)"
    )

    if args.shape:
        c_str, n_str = args.shape.split(",")
        shapes = [(int(c_str), int(n_str))]
    else:
        shapes = parse_shape_file(args.shapes_file)
        print(f"# shapes file : {args.shapes_file}")

    skip_c = {int(x) for x in args.skip_c.split(",") if x.strip()}
    if skip_c:
        before = len(shapes)
        shapes = [(C, N) for (C, N) in shapes if C not in skip_c]
        print(f"# skip C      : {sorted(skip_c)}  ({before - len(shapes)} shapes dropped)")

    print(f"# test points : {len(shapes)}")
    print(f"# quick mode  : {args.quick}")
    print()

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    csv_writer = None
    if csv_file is not None:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "C",
                "num_tokens",
                "warps_m",
                "split_cols",
                "kernel_cfg",
                "occupancy",
                "bpl",
                "ctas_per_row",
                "time_us",
                "max_err",
                "status",
            ]
        )

    best_per_shape: List[
        Tuple[int, int, Tuple[int, int, int, int, int], float, float]
    ] = []

    for C, num_tokens in shapes:
        print(f"=== C={C}, num_tokens={num_tokens} ===")
        torch.manual_seed(0)
        x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device=device)
        weight = torch.empty(C, dtype=torch.bfloat16, device=device).uniform_(
            0.5, 1.5
        )
        ref_out = reference_rmsnorm_silu(x, weight)

        cands = enumerate_candidates(C, args.quick)
        print(f"  candidates: {len(cands)}")

        results: List[
            Tuple[Tuple[int, int, int, int, int], float, float]
        ] = []  # (knobs, time_us, err)

        t_start = time.time()
        best_time = float("inf")
        best_knobs: Optional[Tuple[int, int, int, int, int]] = None
        for i, knobs in enumerate(cands):
            us, err, errmsg = bench_one_candidate(
                C,
                num_tokens,
                knobs,
                x,
                weight,
                ref_out,
                sm_count,
                args.warmup,
                args.iters,
                args.err_tol,
            )
            ctas_per_row = _estimate_ctas_per_row(C, knobs[1], knobs[2], knobs[4])
            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        C,
                        num_tokens,
                        knobs[0],
                        knobs[1],
                        knobs[2],
                        knobs[3],
                        knobs[4],
                        ctas_per_row,
                        f"{us:.3f}" if us is not None else "",
                        f"{err:.4e}" if err is not None else "",
                        errmsg or "ok",
                    ]
                )
                csv_file.flush()
            if us is None:
                continue
            results.append((knobs, us, err))
            if us < best_time:
                best_time = us
                best_knobs = knobs
                print(
                    f"  [{i+1:>3}/{len(cands)}] new best: {knobs} "
                    f"-> {us:8.2f} us  (err={err:.2e})"
                )

        elapsed = time.time() - t_start
        if best_knobs is None:
            print(f"  NO VALID KNOBS FOUND  ({elapsed:.1f}s)")
            print()
            continue

        results.sort(key=lambda r: r[1])
        print(f"  top {min(args.top_k, len(results))} (of {len(results)} valid):")
        for knobs, us, err in results[: args.top_k]:
            print(f"    {knobs}  {us:8.2f} us  err={err:.2e}")
        print(f"  sweep time: {elapsed:.1f}s")
        print()

        best_per_shape.append((C, num_tokens, best_knobs, best_time, results[0][2]))

    if csv_file is not None:
        csv_file.close()

    # Final paste-ready output.
    print()
    print("# === Paste into _KNOB_LUT in flashinfer/jit/rmsnorm_silu.py ===")
    print()
    by_C: dict = {}
    for C, n, knobs, _us, _err in best_per_shape:
        by_C.setdefault(C, []).append((n, knobs))
    for C in sorted(by_C):
        print(f"    # C={C}  (autotuned for WAN VAE shapes)")
        for n, knobs in sorted(by_C[C]):
            print(f'    ({C}, {n}, "bf16"): {knobs},')

    # Range-based LUT: per C, each anchor owns the [lo, hi) interval whose
    # boundaries are the geomean of adjacent anchors. The first range extends
    # down to 0; the last uses 2**31 as an "unbounded upper" sentinel.
    print()
    print("# === Range-based LUT (token_lo <= num_tokens < token_hi) ===")
    print()
    SENTINEL_HI = 2**31
    for C in sorted(by_C):
        items = sorted(by_C[C])
        Ns = [n for n, _ in items]
        print(f"    # C={C}  (anchors={Ns})")
        for i, (N, knobs) in enumerate(items):
            lo = 0 if i == 0 else int((Ns[i - 1] * Ns[i]) ** 0.5)
            hi = (
                SENTINEL_HI
                if i == len(items) - 1
                else int((Ns[i] * Ns[i + 1]) ** 0.5)
            )
            print(f'    ({C}, {lo}, {hi}, "bf16"): {knobs},  # anchor={N}')


if __name__ == "__main__":
    main()
