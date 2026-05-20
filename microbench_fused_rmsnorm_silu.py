#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-shape microbenchmark for the Wan VAE norm+silu replacement.

For each (B,C,T,H,W) shape that hits the fused path during a Wan VAE decode,
times:
  - REF:   WanRMS_norm(x) followed by SiLU (the original path)
  - FUSED: _fused_rmsnorm_silu_5d (flashinfer.norm.fused_rmsnorm_silu)

Inputs are placed in channels_last_3d, matching what test_autoencoder_kl_wan.py
does to the loaded VAE. The shapes file is the 3-column format:
``num_tokens  hidden_size  shape_BxCxTxHxW``.

Usage:
    python microbench_fused_rmsnorm_silu.py
    python microbench_fused_rmsnorm_silu.py --shapes-file path/to/shapes.txt
    python microbench_fused_rmsnorm_silu.py --iters 200 --warmup 20
"""

import argparse
from typing import List, Tuple

import torch


def parse_shape_file(path: str) -> List[Tuple[int, int, int, int, int]]:
    """Parse a 3-column shapes file: num_tokens  hidden_size  shape_BxCxTxHxW.

    Returns a list of (B,C,T,H,W) shapes in file order.
    """
    out: List[Tuple[int, ...]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) != 3:
                continue
            shape_str = parts[2].strip()
            shape = tuple(int(x.strip()) for x in shape_str.strip("()").split(","))
            if len(shape) != 5:
                raise ValueError(f"expected 5D shape, got {shape!r} in line {line!r}")
            out.append(shape)
    return out


def bench_one(
    shape: Tuple[int, int, int, int, int],
    dtype: torch.dtype,
    device: str,
    warmup: int,
    iters: int,
):
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanRMS_norm,
        _fused_rmsnorm_silu_5d,
    )

    B, C, T, H, W = shape
    x = torch.randn(*shape, dtype=dtype, device=device).contiguous(
        memory_format=torch.channels_last_3d
    )
    norm = WanRMS_norm(C, images=False).to(device).to(dtype)
    silu = torch.nn.SiLU()
    with torch.no_grad():
        # randomize gamma so the kernel does real work and doesn't optimize to a no-op
        norm.gamma.uniform_(0.5, 1.5)

    # warmup both paths
    for _ in range(warmup):
        _ = silu(norm(x))
        _ = _fused_rmsnorm_silu_5d(x, norm.gamma)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # REF
    start.record()
    for _ in range(iters):
        y_ref = silu(norm(x))
    end.record()
    torch.cuda.synchronize()
    ref_us = start.elapsed_time(end) / iters * 1000.0

    # FUSED
    start.record()
    for _ in range(iters):
        y_fused = _fused_rmsnorm_silu_5d(x, norm.gamma)
    end.record()
    torch.cuda.synchronize()
    fused_us = start.elapsed_time(end) / iters * 1000.0

    # numerical diff between final outputs of each path
    d = (y_ref.float() - y_fused.float()).abs()
    return ref_us, fused_us, d.max().item(), d.mean().item()


def parse_dtype(s: str) -> torch.dtype:
    m = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = s.lower()
    if key not in m:
        raise ValueError(f"unsupported dtype: {s}")
    return m[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shapes-file",
        default="wan_vae_fused_rmsnorm_silu_shapes_from_image.txt",
        help="3-column shapes file: num_tokens  hidden_size  shape_BxCxTxHxW",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dtype = parse_dtype(args.dtype)
    shapes = parse_shape_file(args.shapes_file)

    print(f"# shapes file : {args.shapes_file}")
    print(f"# dtype       : {dtype}")
    print(f"# warmup={args.warmup}  iters={args.iters}")
    print()
    header = (
        f"{'shape (B,C,T,H,W)':<28} {'tokens':>10} {'C':>5} "
        f"{'ref_us':>10} {'fused_us':>10} {'speedup':>9} {'max_abs':>10}"
    )
    print(header)
    print("-" * len(header))

    total_ref_us = 0.0
    total_fused_us = 0.0
    for shape in shapes:
        B, C, T, H, W = shape
        try:
            ref_us, fused_us, max_d, _ = bench_one(
                shape, dtype, args.device, args.warmup, args.iters
            )
        except Exception as e:
            print(
                f"{str(shape):<28} {B*T*H*W:>10} {C:>5} "
                f"  SKIP: {type(e).__name__}: {e}"
            )
            continue
        total_ref_us += ref_us
        total_fused_us += fused_us
        speedup = ref_us / fused_us if fused_us > 0 else float("inf")
        print(
            f"{str(shape):<28} {B*T*H*W:>10} {C:>5} "
            f"{ref_us:>10.2f} {fused_us:>10.2f} {speedup:>8.2f}x {max_d:>10.2e}"
        )

    print()
    print(
        f"total (one call per shape):  "
        f"ref={total_ref_us/1000:.2f} ms  fused={total_fused_us/1000:.2f} ms"
    )
    if total_fused_us > 0:
        print(
            f"aggregate speedup: {total_ref_us/total_fused_us:.3f}x  "
            f"(delta {(total_ref_us - total_fused_us)/1000:+.2f} ms)"
        )


if __name__ == "__main__":
    main()
