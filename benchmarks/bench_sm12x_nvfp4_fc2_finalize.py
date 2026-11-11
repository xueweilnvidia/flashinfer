"""Paired SM120 NVFP4 FC2 (down projection + weighted finalize) configuration benchmark.

Compares FC2 launch configurations of ``cute_dsl_sm12x_fc2_finalize_nvfp4`` on
the same inputs, for several token counts:

    python benchmarks/bench_sm12x_nvfp4_fc2_finalize.py
    python benchmarks/bench_sm12x_nvfp4_fc2_finalize.py --num-tokens 16384 \
        --reference-tokens 0 --output fc2.json

Configurations (``--configs``):
    default    tile 128x128x128 with no FC2 epilogue options (current default path)
    optimized  tile 128x128x128, swizzle_size=8, separate_epilogue=True, store_warps=2

Shape: H=4096, I=768, E=384, top-k=20; each token independently selects 20
distinct random experts (``--seed``). FC2 inputs are the real packed FC1 Q1
outputs (SwiGLU alpha=1, beta=0, limit=20); FC1, routing and weight preparation
are outside timing.

Timing: one CUDA graph per configuration; output reset and a 2x-L2 flush are
outside each timed replay (cold L2); configurations are interleaved in shuffled
order for ``--rounds`` rounds of ``--iters`` replays, and the median of the
round medians is reported. Clocks are not locked, so compare configurations
only within one run.

Correctness: every configuration is checked eagerly and after every round
against an independent Torch oracle built from the actual packed FC2 inputs
(``--reference-tokens 0`` checks every token). The oracle models the kernel's
BF16 staging; BF16 atomic accumulation order is nondeterministic, so the check
uses relative L2 (typically ~5.5e-3) rather than bitwise equality.

The ``default`` path shares the optimized mainloop, so ``optimized/default``
is slightly lower than a comparison against the kernel before the FC2
optimizations.
"""

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 4096, 768, 384, 20
TILE = (128, 128, 128)
CONFIGS = {
    "default": dict(),
    "optimized": dict(swizzle_size=8, separate_epilogue=True, store_warps=2),
}


@dataclass
class Fc2Inputs:
    q: torch.Tensor
    sf: torch.Tensor
    weight: torch.Tensor
    weight_sf: torch.Tensor
    offsets: torch.Tensor
    token_map: torch.Tensor
    route_weights: torch.Tensor
    weight_global: torch.Tensor
    input_global: torch.Tensor
    out: torch.Tensor
    reference_weight: torch.Tensor
    reference_weight_sf: torch.Tensor
    routing_sha256: str


def prepare(num_tokens, seed, device):
    from routines.unified_moe import _canonical_inputs, _quantize_cutile_nvfp4_source

    from flashinfer.fused_moe import SM12xNvfp4Config, SwiGLU
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_fp4_fc1_act_q1 as fc1

    activation = SwiGLU(alpha=1, beta=0, limit=20)
    config = SimpleNamespace(
        num_tokens=num_tokens,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=EXPERTS,
        top_k=TOP_K,
        random_seed=seed,
        routed_scaling_factor=1.0,
    )
    act, w1, w2 = _canonical_inputs(config, activation, device)
    act.topk_ids = (
        torch.rand(num_tokens, EXPERTS, device=device)
        .argsort(1)[:, :TOP_K]
        .to(torch.int32)
        .contiguous()
    )
    act.hidden_states_q.mul_(16)
    routing_sha256 = hashlib.sha256(act.topk_ids.cpu().numpy().tobytes()).hexdigest()
    q1, s1, g1 = _quantize_cutile_nvfp4_source(w1)
    q2, s2, g2 = _quantize_cutile_nvfp4_source(w2)
    view = SM12xNvfp4Config.prepare_weights(
        q1,
        s1,
        g1,
        q2,
        s2,
        g2,
        num_local_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        activation=activation,
        device=device,
    )
    one = torch.ones(1, dtype=torch.float32, device=device)
    offsets, tokens, scores, q, sf = fc1.cute_dsl_sm12x_route_fc1_act_q1_nvfp4(
        act.hidden_states_q,
        act.topk_ids,
        act.topk_weights,
        view["w1"],
        view["w1_scale"],
        one,
        view["w1_global_scale"],
        one,
        tile=TILE,
        enable_pdl=False,
        swiglu_limit=20,
    )
    torch.cuda.synchronize()
    return Fc2Inputs(
        q,
        sf,
        view["w2"],
        view["w2_scale"],
        offsets,
        tokens,
        scores,
        view["w2_global_scale"],
        one,
        torch.empty(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16),
        q2,
        s2,
        routing_sha256,
    )


def run_fc2(data, options):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp4_fc2_finalize import (
        cute_dsl_sm12x_fc2_finalize_nvfp4,
    )

    return cute_dsl_sm12x_fc2_finalize_nvfp4(
        data.q,
        data.sf,
        data.weight,
        data.weight_sf,
        data.offsets,
        data.token_map,
        data.route_weights,
        data.weight_global,
        data.input_global,
        data.out.shape[0],
        tile=TILE,
        tune=False,
        enable_pdl=False,
        out=data.out,
        **options,
    )


def _dequant(payload, scales):
    lut = payload.new_tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    codes = torch.stack((payload & 15, payload >> 4), -1).flatten(-2).long()
    return lut[codes] * scales.float().repeat_interleave(16, dim=-1)


def _unpack_q1_scales(sf, rows, columns):
    """Read selected physical rows from the 128x4 E4M3 scale swizzle."""
    row = rows.long()[:, None]
    col = torch.arange(columns, device=sf.device)[None, :]
    padded_columns = (columns + 3) // 4 * 4
    index = (
        ((row >> 7) * (padded_columns >> 2) + (col >> 2)) * 512
        + (row & 31) * 16
        + ((row & 127) >> 5) * 4
        + (col & 3)
    )
    return sf.flatten()[index].view(torch.float8_e4m3fn)


def reference(data, requested_tokens):
    """Torch oracle from the ACTUAL packed FC2 inputs for a token sample."""
    m = data.out.shape[0]
    count = m if requested_tokens == 0 else min(requested_tokens, m)
    sample = torch.linspace(0, m - 1, count, device=data.q.device).long()
    inverse = torch.full((m,), -1, device=data.q.device, dtype=torch.long)
    inverse[sample] = torch.arange(count, device=data.q.device)
    result = torch.zeros(count, HIDDEN, device=data.q.device, dtype=torch.float32)
    offsets = data.offsets.cpu().tolist()
    tokens = data.token_map.long()
    contributions = 0
    for expert, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        sample_rows = inverse[tokens[begin:end]]
        local = torch.where(sample_rows >= 0)[0]
        if local.numel() == 0:
            continue
        contributions += local.numel()
        packed_rows = begin + local
        # Q1 payload rows are compact; scale rows have per-expert padding.
        sf_base = (begin + expert * 127) // 128 * 128
        scales = _unpack_q1_scales(data.sf, sf_base + local, INTERMEDIATE // 16)
        a = _dequant(data.q[packed_rows], scales) / data.input_global
        b = _dequant(data.reference_weight[expert], data.reference_weight_sf[expert])
        projection = (a @ b.T) * data.weight_global[expert]
        # Match FC2's BF16 staging and BF16 contribution before atomic reduction.
        partial = (
            (
                projection.to(torch.bfloat16).float()
                * data.route_weights[packed_rows, None]
            )
            .to(torch.bfloat16)
            .float()
        )
        result.index_add_(0, sample_rows[local], partial)
    assert contributions == count * TOP_K, (contributions, count * TOP_K)
    return sample, result


def validate(data, sample, expected, label):
    actual = data.out[sample].float()
    record = dict(
        relative_l2=float(
            (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        ),
        max_abs_error=float((actual - expected).abs().max()),
        finite=bool(torch.isfinite(data.out).all()),
        sampled_tokens=sample.numel(),
    )
    # FP32 oracle reduction vs nondeterministic BF16 atomic accumulation.
    if not record["finite"] or record["relative_l2"] > 0.02:
        raise RuntimeError(f"{label}: FC2 correctness check failed: {record}")
    return record


def bench_tokens(args, num_tokens, device):
    data = prepare(num_tokens, args.seed, device)
    sample, expected = reference(data, args.reference_tokens)
    graphs, records = [], []
    for name in args.configs:
        options = CONFIGS[name]
        data.out.zero_()
        run_fc2(data, options)
        torch.cuda.synchronize()
        eager = validate(data, sample, expected, name)
        for _ in range(args.warmup):
            data.out.zero_()
            run_fc2(data, options)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        data.out.zero_()
        with torch.cuda.graph(graph):
            run_fc2(data, options)
        graphs.append(graph)
        records.append(
            dict(config=name, options=options, eager_check=eager, round_medians_ms=[])
        )

    props = torch.cuda.get_device_properties(device)
    flush = torch.empty(2 * props.L2_cache_size, device=device, dtype=torch.uint8)
    for _ in range(args.settle_iters):
        data.out.zero_()
        flush.zero_()
        graphs[0].replay()
    torch.cuda.synchronize()
    for rnd in range(args.rounds):
        order = list(range(len(graphs)))
        random.Random(args.seed + rnd).shuffle(order)
        for idx in order:
            for _ in range(args.warmup):
                data.out.zero_()
                graphs[idx].replay()
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(args.iters)
            ]
            for start, end in events:
                data.out.zero_()
                flush.zero_()
                start.record()
                graphs[idx].replay()
                end.record()
            torch.cuda.synchronize()
            records[idx]["round_medians_ms"].append(
                statistics.median(s.elapsed_time(e) for s, e in events)
            )
            records[idx]["replay_check"] = validate(
                data, sample, expected, records[idx]["config"]
            )
    for record in records:
        record["median_ms"] = statistics.median(record["round_medians_ms"])
    result = dict(
        num_tokens=num_tokens,
        routing_sha256=data.routing_sha256,
        records=records,
    )
    del data, graphs, flush
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=[4000, 6000, 9000, 16384]
    )
    parser.add_argument(
        "--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--settle-iters",
        type=int,
        default=500,
        help="replays before timing so GPU clocks settle",
    )
    parser.add_argument(
        "--reference-tokens",
        type=int,
        default=256,
        help="tokens checked against the Torch oracle; 0 checks every token",
    )
    parser.add_argument("--output", type=Path, help="write raw results as JSON")
    args = parser.parse_args()
    if min(min(args.num_tokens), args.rounds, args.iters) <= 0 or min(
        args.warmup, args.settle_iters, args.reference_tokens
    ) < 0:
        parser.error("token counts/rounds/iters must be positive, others nonnegative")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] != 12:
        raise SystemExit("requires an SM12x GPU")

    results = []
    for num_tokens in args.num_tokens:
        result = bench_tokens(args, num_tokens, device)
        results.append(result)
        base = result["records"][0]["median_ms"]
        for record in result["records"]:
            check = record["replay_check"]
            print(
                f"tokens={num_tokens:<6} {record['config']:<10} "
                f"{record['median_ms']:.4f} ms  x{base / record['median_ms']:.3f}  "
                f"rel_l2={check['relative_l2']:.6f}",
                flush=True,
            )

    print(f"\n| tokens | {' | '.join(f'{c} (ms)' for c in args.configs)} | speedup |")
    print("|---|" + "---|" * (len(args.configs) + 1))
    for result in results:
        times = [r["median_ms"] for r in result["records"]]
        print(
            f"| {result['num_tokens']} | "
            + " | ".join(f"{t:.4f}" for t in times)
            + f" | {times[0] / times[-1]:.3f}x |"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    gpu=torch.cuda.get_device_name(device),
                    shape=dict(H=HIDDEN, I=INTERMEDIATE, E=EXPERTS, top_k=TOP_K),
                    tile=TILE,
                    seed=args.seed,
                    rounds=args.rounds,
                    iters=args.iters,
                    reference_tokens=args.reference_tokens,
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
