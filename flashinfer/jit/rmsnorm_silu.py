"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import bisect
import shutil

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import write_if_different


# Sweep-tuned point LUT for Blackwell (SM100 B200, SM103 B300) LLM-shape
# problem sizes. Format: (warps_m, split_cols, kernel_cfg, occupancy,
# bytes_per_ldg). Non-power-of-2 hidden sizes (C in {96, 192, 384}) used by
# Wan2.2 VAE decoder shapes are covered by the range LUT below.
_KNOB_LUT = {
    # C=64
    (64, 1560, "bf16"): (8, 0, 0, 2, 4),
    (64, 6240, "bf16"): (32, 4, 0, 2, 4),
    (64, 24960, "bf16"): (32, 4, 0, 2, 4),
    (64, 99840, "bf16"): (8, 0, 1, 8, 4),
    (64, 399360, "bf16"): (4, 0, 1, 16, 4),
    (64, 1560, "fp8"): (8, 4, 0, 6, 4),
    (64, 6240, "fp8"): (8, 0, 0, 3, 2),
    (64, 24960, "fp8"): (8, 0, 0, 7, 4),
    (64, 99840, "fp8"): (8, 0, 1, 6, 2),
    (64, 399360, "fp8"): (32, 0, 1, 2, 2),
    (64, 1560, "nvfp4"): (8, 0, 2, 1, 4),
    (64, 6240, "nvfp4"): (8, 4, 0, 4, 4),
    (64, 24960, "nvfp4"): (8, 0, 1, 6, 4),
    (64, 99840, "nvfp4"): (32, 4, 1, 2, 4),
    (64, 399360, "nvfp4"): (32, 4, 1, 2, 4),
    # C=128
    (128, 1560, "bf16"): (8, 4, 0, 3, 4),
    (128, 6240, "bf16"): (8, 0, 0, 3, 4),
    (128, 24960, "bf16"): (8, 0, 0, 6, 4),
    (128, 99840, "bf16"): (32, 4, 0, 2, 4),
    (128, 399360, "bf16"): (8, 0, 0, 8, 4),
    (128, 1560, "fp8"): (8, 0, 0, 3, 4),
    (128, 6240, "fp8"): (8, 0, 0, 4, 8),
    (128, 24960, "fp8"): (8, 0, 0, 8, 8),
    (128, 99840, "fp8"): (32, 0, 0, 2, 8),
    (128, 399360, "fp8"): (32, 0, 0, 2, 8),
    (128, 1560, "nvfp4"): (8, 4, 0, 3, 8),
    (128, 6240, "nvfp4"): (8, 0, 0, 5, 8),
    (128, 24960, "nvfp4"): (8, 0, 1, 8, 8),
    (128, 99840, "nvfp4"): (32, 0, 1, 2, 8),
    (128, 399360, "nvfp4"): (32, 0, 1, 2, 8),
    # C=160
    (160, 1560, "bf16"): (8, 0, 0, 4, 2),
    (160, 6240, "bf16"): (8, 0, 0, 4, 2),
    (160, 24960, "bf16"): (8, 4, 1, 6, 2),
    (160, 99840, "bf16"): (32, 4, 1, 2, 2),
    (160, 399360, "bf16"): (32, 4, 1, 2, 2),
    (160, 1560, "fp8"): (8, 0, 0, 2, 2),
    (160, 6240, "fp8"): (8, 0, 0, 4, 2),
    (160, 24960, "fp8"): (8, 4, 0, 6, 2),
    (160, 99840, "fp8"): (32, 4, 1, 2, 2),
    (160, 399360, "fp8"): (32, 4, 1, 2, 2),
    (160, 1560, "nvfp4"): (4, 4, 0, 4, 2),
    (160, 6240, "nvfp4"): (8, 0, 1, 4, 2),
    (160, 24960, "nvfp4"): (8, 4, 1, 8, 2),
    (160, 99840, "nvfp4"): (32, 4, 0, 1, 2),
    (160, 399360, "nvfp4"): (32, 0, 1, 2, 2),
    # C=256
    (256, 1560, "bf16"): (8, 0, 0, 6, 16),
    (256, 6240, "bf16"): (8, 0, 0, 4, 4),
    (256, 24960, "bf16"): (8, 0, 0, 8, 16),
    (256, 99840, "bf16"): (4, 4, 0, 16, 16),
    (256, 399360, "bf16"): (4, 0, 0, 16, 16),
    (256, 1560, "fp8"): (8, 4, 0, 2, 4),
    (256, 6240, "fp8"): (8, 0, 0, 4, 4),
    (256, 24960, "fp8"): (8, 4, 0, 8, 16),
    (256, 99840, "fp8"): (4, 0, 0, 16, 16),
    (256, 399360, "fp8"): (32, 0, 0, 2, 16),
    (256, 1560, "nvfp4"): (8, 0, 2, 1, 16),
    (256, 6240, "nvfp4"): (8, 0, 2, 1, 16),
    (256, 24960, "nvfp4"): (8, 4, 1, 6, 16),
    (256, 99840, "nvfp4"): (32, 0, 1, 1, 16),
    (256, 399360, "nvfp4"): (32, 0, 1, 2, 16),
    # C=320
    (320, 1560, "bf16"): (8, 4, 1, 4, 4),
    (320, 6240, "bf16"): (8, 4, 0, 5, 4),
    (320, 24960, "bf16"): (8, 0, 0, 5, 4),
    (320, 99840, "bf16"): (4, 0, 1, 16, 4),
    (320, 399360, "bf16"): (32, 4, 0, 2, 4),
    (320, 1560, "fp8"): (8, 0, 0, 2, 4),
    (320, 6240, "fp8"): (8, 0, 0, 5, 4),
    (320, 24960, "fp8"): (8, 0, 0, 5, 4),
    (320, 99840, "fp8"): (32, 0, 1, 2, 4),
    (320, 399360, "fp8"): (32, 0, 1, 2, 4),
    (320, 1560, "nvfp4"): (4, 4, 0, 9, 4),
    (320, 6240, "nvfp4"): (4, 0, 0, 9, 4),
    (320, 24960, "nvfp4"): (8, 0, 1, 8, 4),
    (320, 99840, "nvfp4"): (32, 4, 1, 2, 4),
    (320, 399360, "nvfp4"): (32, 4, 1, 2, 4),
    # C=512
    (512, 1560, "bf16"): (8, 0, 0, 2, 16),
    (512, 6240, "bf16"): (8, 0, 0, 5, 16),
    (512, 24960, "bf16"): (4, 0, 0, 8, 16),
    (512, 99840, "bf16"): (4, 0, 2, 1, 8),
    (512, 399360, "bf16"): (4, 0, 2, 1, 4),
    (512, 1560, "fp8"): (8, 0, 0, 2, 8),
    (512, 6240, "fp8"): (8, 0, 0, 4, 8),
    (512, 24960, "fp8"): (4, 0, 0, 9, 8),
    (512, 99840, "fp8"): (32, 4, 1, 2, 8),
    (512, 399360, "fp8"): (32, 4, 1, 2, 8),
    (512, 1560, "nvfp4"): (4, 4, 0, 3, 16),
    (512, 6240, "nvfp4"): (4, 0, 0, 9, 16),
    (512, 24960, "nvfp4"): (4, 0, 2, 1, 16),
    (512, 99840, "nvfp4"): (32, 4, 0, 1, 16),
    (512, 399360, "nvfp4"): (32, 0, 0, 1, 16),
    # C=640
    (640, 1560, "bf16"): (4, 0, 0, 4, 4),
    (640, 6240, "bf16"): (4, 0, 0, 5, 4),
    (640, 24960, "bf16"): (4, 0, 0, 5, 4),
    (640, 99840, "bf16"): (4, 0, 2, 1, 8),
    (640, 399360, "bf16"): (4, 0, 2, 1, 8),
    (640, 1560, "fp8"): (4, 0, 0, 3, 8),
    (640, 6240, "fp8"): (8, 0, 0, 4, 8),
    (640, 24960, "fp8"): (8, 0, 0, 4, 8),
    (640, 99840, "fp8"): (4, 4, 0, 9, 8),
    (640, 399360, "fp8"): (32, 4, 1, 2, 8),
    (640, 1560, "nvfp4"): (4, 4, 0, 5, 8),
    (640, 6240, "nvfp4"): (4, 0, 1, 9, 8),
    (640, 24960, "nvfp4"): (4, 0, 2, 1, 8),
    (640, 99840, "nvfp4"): (32, 0, 1, 1, 8),
    (640, 399360, "nvfp4"): (32, 4, 1, 1, 8),
    # C=1024
    (1024, 1560, "bf16"): (4, 4, 0, 3, 16),
    (1024, 6240, "bf16"): (4, 0, 0, 5, 16),
    (1024, 24960, "bf16"): (4, 4, 1, 10, 16),
    (1024, 99840, "bf16"): (8, 0, 2, 1, 16),
    (1024, 399360, "bf16"): (8, 0, 2, 1, 16),
    (1024, 1560, "fp8"): (4, 0, 0, 3, 4),
    (1024, 6240, "fp8"): (4, 0, 0, 5, 8),
    (1024, 24960, "fp8"): (1, 4, 0, 16, 8),
    (1024, 99840, "fp8"): (4, 0, 1, 9, 8),
    (1024, 399360, "fp8"): (32, 4, 1, 1, 8),
    (1024, 1560, "nvfp4"): (4, 4, 0, 7, 16),
    (1024, 6240, "nvfp4"): (4, 0, 2, 1, 16),
    (1024, 24960, "nvfp4"): (4, 0, 2, 1, 16),
    (1024, 99840, "nvfp4"): (32, 0, 1, 1, 16),
    (1024, 399360, "nvfp4"): (32, 4, 1, 1, 16),
}

# Range-based knob LUT. Key is (C, token_lo, token_hi, dtype) with half-open
# interval token_lo <= num_tokens < token_hi. Each range maps to the knobs of
# a single autotuned "anchor" shape (logged in the trailing comment). Generated
# by sweep_fused_rmsnorm_silu_knobs.py with TP divisors (1, 2, 4, 8); range
# boundaries are the geometric means of adjacent anchors so that any token
# count projects to its nearest anchor on a log scale.
_KNOB_LUT_RANGES = {
    # C=96  (anchors=[66560, 133120, 266240, 532480, 1064960, 2129920])
    (96, 0, 94130, "bf16"): (8, 0, 1, 16, 2),  # anchor=66560
    (96, 94130, 188260, "bf16"): (8, 0, 1, 16, 2),  # anchor=133120
    (96, 188260, 376520, "bf16"): (32, 0, 1, 2, 2),  # anchor=266240
    (96, 376520, 753040, "bf16"): (32, 0, 1, 2, 2),  # anchor=532480
    (96, 753040, 1506080, "bf16"): (32, 0, 1, 2, 2),  # anchor=1064960
    (96, 1506080, 2147483648, "bf16"): (32, 0, 1, 2, 2),  # anchor=2129920
    # C=192 (anchors=[4160, 8320, 16640, 33280, 66560, 133120, 266240, 532480])
    (192, 0, 5883, "bf16"): (32, 0, 0, 1, 4),  # anchor=4160
    (192, 5883, 11766, "bf16"): (32, 0, 0, 1, 4),  # anchor=8320
    (192, 11766, 23532, "bf16"): (8, 0, 0, 9, 4),  # anchor=16640
    (192, 23532, 47065, "bf16"): (8, 0, 1, 6, 4),  # anchor=33280
    (192, 47065, 94130, "bf16"): (8, 0, 1, 16, 4),  # anchor=66560
    (192, 94130, 188260, "bf16"): (32, 0, 0, 2, 4),  # anchor=133120
    (192, 188260, 376520, "bf16"): (32, 0, 1, 2, 4),  # anchor=266240
    (192, 376520, 2147483648, "bf16"): (32, 0, 1, 2, 4),  # anchor=532480
    # C=384 (anchors=[1040, 2080, 4160, 8320, 16640, 33280, 66560])
    (384, 0, 1470, "bf16"): (32, 0, 0, 1, 8),  # anchor=1040
    (384, 1470, 2941, "bf16"): (32, 0, 0, 16, 4),  # anchor=2080
    (384, 2941, 5883, "bf16"): (8, 0, 2, 4, 8),  # anchor=4160
    (384, 5883, 11766, "bf16"): (32, 0, 2, 6, 4),  # anchor=8320
    (384, 11766, 23532, "bf16"): (8, 0, 0, 4, 4),  # anchor=16640
    (384, 23532, 47065, "bf16"): (8, 0, 0, 10, 4),  # anchor=33280
    (384, 47065, 2147483648, "bf16"): (8, 0, 1, 16, 4),  # anchor=66560
}


def _build_range_index(lut):
    """Bucket range entries by (C, dtype) and presort by lo for bisect lookup."""
    by_key = {}
    for (C, lo, hi, dtype), knobs in lut.items():
        by_key.setdefault((C, dtype), []).append((lo, hi, knobs))
    index = {}
    for k, entries in by_key.items():
        entries.sort(key=lambda x: x[0])
        los = [e[0] for e in entries]
        index[k] = (los, entries)
    return index


_KNOB_LUT_RANGES_INDEX = _build_range_index(_KNOB_LUT_RANGES)

# Cartesian product enumerated by AOT to pre-compile every LUT-hit module.
# Pairs absent from the LUTs fall back to _compute_default_knobs and dedupe by
# knob URI (the fallback is independent of num_tokens).
_SUPPORTED_C = [64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 1024]
_SUPPORTED_TOKENS = [
    # LLM-shape token counts (covered by point LUT)
    1560, 6240, 24960, 99840, 399360,
    # Wan2.2 VAE anchors (covered by range LUT for C in {96, 192, 384})
    1040, 2080, 4160, 8320, 16640, 33280, 66560, 133120, 266240, 532480,
    1064960, 2129920,
]


def _compute_default_knobs(C: int, dtype: str):
    """Conservative fallback knobs for non-LUT sizes."""
    input_size = 2  # bf16
    warps_m = 32 if dtype == "nvfp4" else 1
    warps_n = 1
    cpr = 1

    for bpl in [4, 8, 16, 2]:
        num_elts = bpl // input_size
        if num_elts <= 0 or C % num_elts != 0:
            continue
        vec_cols = C // num_elts
        vec_cols_per_ldg = cpr * warps_n * 32
        if vec_cols_per_ldg <= 0 or vec_cols % vec_cols_per_ldg != 0:
            continue
        ldgs = vec_cols // vec_cols_per_ldg
        if ldgs > 1024:
            continue
        return (warps_m, 0, 0, 1, bpl)

    return None


def select_knobs(C: int, num_tokens: int, dtype: str, sm_version: int = 100):
    """Select knobs from LUT or fallback heuristic.

    Lookup order on SM100+:
      1. Exact-point LUT (``_KNOB_LUT``) keyed by ``(C, num_tokens, dtype)``.
      2. Range LUT (``_KNOB_LUT_RANGES``) keyed by
         ``(C, token_lo, token_hi, dtype)``, matched when
         ``token_lo <= num_tokens < token_hi``.
      3. Conservative fallback via ``_compute_default_knobs``.

    Non-SM100 (or no match anywhere) falls through to the fallback.
    """
    if sm_version >= 100:
        point = _KNOB_LUT.get((C, num_tokens, dtype))
        if point is not None:
            return point
        bucket = _KNOB_LUT_RANGES_INDEX.get((C, dtype))
        if bucket is not None:
            los, entries = bucket
            i = bisect.bisect_right(los, num_tokens) - 1
            if i >= 0:
                lo, hi, knobs = entries[i]
                if lo <= num_tokens < hi:
                    return knobs
    return _compute_default_knobs(C, dtype)


def _estimate_ctas_per_row(
    C: int, split_cols: int, kernel_cfg: int, bytes_per_ldg: int, warps_n: int = 1
) -> int:
    """Estimate CTAS_PER_ROW from knobs."""
    if split_cols != 4 or kernel_cfg == 2:
        return 1
    input_size = 2  # bf16
    num_elts = bytes_per_ldg // input_size
    elts_per_ldg = num_elts * warps_n * 32
    if elts_per_ldg <= 0 or C % elts_per_ldg != 0:
        return 1
    ldgs_per_row = C // elts_per_ldg
    ldgs_to_cause_register_spill = 64 // num_elts if num_elts > 0 else 1
    ctas_per_row = 1
    for ldgs in range(min(ldgs_per_row, ldgs_to_cause_register_spill - 1), 0, -1):
        if ldgs_per_row % ldgs == 0:
            ctas_per_row = ldgs_per_row // ldgs
            break
    return ctas_per_row


def _generate_config(
    C: int,
    output_dtype: str,
    warps_m: int,
    ctas_per_row: int,
    bytes_per_ldg: int,
    kernel_cfg: int,
    occupancy: int,
) -> str:
    """Generate the constexpr config .inc file content."""
    lines = [
        "// Auto-generated RmsNorm+SiLU kernel config. Do not edit.",
        "",
        "using ITYPE = nv_bfloat16;",
    ]

    is_fp8 = output_dtype == "fp8"
    is_nvfp4 = output_dtype == "nvfp4"

    if output_dtype == "bf16":
        lines.append("using OTYPE = nv_bfloat16;")
        lines.append("using NORM_OTYPE = nv_bfloat16;")
    elif output_dtype == "fp8":
        lines.append("using OTYPE = nv_fp8_e4m3;")
        lines.append("using NORM_OTYPE = float;")
    elif output_dtype == "nvfp4":
        lines.append("using OTYPE = nv_fp4_e2m1;")
        lines.append("using NORM_OTYPE = float;")

    lines += [
        "using WTYPE = nv_bfloat16;",
        "using CTYPE = float;",
        "",
        f"constexpr int HIDDEN_SIZE = {C};",
        "constexpr int BATCH_SIZE = 1;",
        f"constexpr int CTAS_PER_ROW = {ctas_per_row};",
        f"constexpr int WARPS_M = {warps_m};",
        "constexpr int WARPS_N = 1;",
        f"constexpr int BYTES_PER_LDG = {bytes_per_ldg};",
        f"constexpr int KERNEL_CFG = {kernel_cfg};",
        "constexpr bool isRMSNorm = true;",
        "constexpr bool isAdaLN = false;",
        "constexpr bool isBatchFirst = true;",
        "constexpr bool hasGamma = true;",
        "constexpr bool hasBeta = false;",
        "constexpr bool isZeroCenteredGamma = false;",
        "constexpr bool isZeroCenteredGammaCastBeforeAdd = false;",
    ]

    use_smem_gamma = kernel_cfg == 1
    use_non_persistent = kernel_cfg == 2
    lines += [
        f"constexpr bool useSmemGamma = {'true' if use_smem_gamma else 'false'};",
        f"constexpr bool GAMMA_ON_DEMAND = {'true' if (not use_smem_gamma and use_non_persistent) else 'false'};",
        f"constexpr bool isFP8Out = {'true' if is_fp8 else 'false'};",
        "constexpr bool hasScaleInv = false;",
        "constexpr bool hasAmax = false;",
        "#define LN_USE_CLUSTER 0",
        "constexpr bool USE_CLUSTER = false;",
        f"constexpr bool isBlockScaleOut = {'true' if is_nvfp4 else 'false'};",
        f"constexpr bool isFP4Out = {'true' if is_nvfp4 else 'false'};",
        f"constexpr bool isBlockScale_1D1X1X = {'true' if is_nvfp4 else 'false'};",
        "constexpr bool isBlockScale_1D2X2X = false;",
        "constexpr bool isBlockScale_1D2X2X_Transpose = false;",
        "constexpr bool useBlockScaleColwiseKernel = false;",
        f"constexpr int DESIRED_OCCUPANCY = {occupancy};",
        "",
        "using Ktraits = Kernel_traits<WTYPE, ITYPE, OTYPE, CTYPE, NORM_OTYPE, uint32_t,",
        "    HIDDEN_SIZE, BATCH_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG,",
        "    isRMSNorm, isAdaLN, isBatchFirst, hasGamma, hasBeta, useSmemGamma,",
        "    USE_CLUSTER, false>;",
        "",
        "#define USE_STATIC_SMEM_VALUE ((int)sizeof(LnFwdShared<Ktraits>))",
    ]

    return "\n".join(lines) + "\n"


def _get_uri(
    C: int,
    output_dtype: str,
    warps_m: int,
    ctas_per_row: int,
    bytes_per_ldg: int,
    kernel_cfg: int,
    occupancy: int,
) -> str:
    return (
        f"rmsnorm_silu_C{C}_{output_dtype}"
        f"_wm{warps_m}_cpr{ctas_per_row}_bpl{bytes_per_ldg}"
        f"_cfg{kernel_cfg}_occ{occupancy}"
    )


def gen_rmsnorm_silu_module(
    C: int,
    output_dtype: str,
    warps_m: int,
    ctas_per_row: int,
    bytes_per_ldg: int,
    kernel_cfg: int,
    occupancy: int,
) -> JitSpec:
    uri = _get_uri(
        C, output_dtype, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
    )

    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)

    config_content = _generate_config(
        C, output_dtype, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
    )
    write_if_different(gen_directory / "rmsnorm_silu_config.inc", config_content)

    sources = []
    for fname in ["rmsnorm_silu.cu", "flashinfer_rmsnorm_silu_binding.cu"]:
        src = jit_env.FLASHINFER_CSRC_DIR / fname
        dst = gen_directory / fname
        # Preserve dst mtime when content is unchanged so ninja doesn't rebuild
        # every time gen_rmsnorm_silu_module() is called.
        src_bytes = src.read_bytes()
        if not dst.exists() or dst.read_bytes() != src_bytes:
            dst.write_bytes(src_bytes)
        sources.append(dst)

    return gen_jit_spec(
        uri,
        sources,
        extra_cuda_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ],
        extra_include_paths=[str(gen_directory)],
    )
