import argparse
import json
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _triton_edge_projection_kernel(
        state,
        edge,
        time,
        weight,
        bias,
        edge_src,
        edge_to_eid,
        out,
        ROWS,
        STATE_DIM: tl.constexpr,
        EDGE_DIM: tl.constexpr,
        TIME_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        src = tl.load(edge_src + offs_m, mask=offs_m < ROWS, other=0)
        eid = tl.load(edge_to_eid + offs_m, mask=offs_m < ROWS, other=0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        total_dim: tl.constexpr = STATE_DIM + EDGE_DIM + TIME_DIM

        for k0 in range(0, total_dim, BLOCK_K):
            k = k0 + offs_k
            state_mask = k < STATE_DIM
            edge_mask = (k >= STATE_DIM) & (k < STATE_DIM + EDGE_DIM)
            time_mask = k >= STATE_DIM + EDGE_DIM
            state_k = k
            edge_k = k - STATE_DIM
            time_k = k - STATE_DIM - EDGE_DIM

            a_state = tl.load(
                state + src[:, None] * STATE_DIM + state_k[None, :],
                mask=(offs_m[:, None] < ROWS) & state_mask[None, :],
                other=0.0,
            )
            a_edge = tl.load(
                edge + eid[:, None] * EDGE_DIM + edge_k[None, :],
                mask=(offs_m[:, None] < ROWS) & edge_mask[None, :],
                other=0.0,
            )
            a_time = tl.load(
                time + offs_m[:, None] * TIME_DIM + time_k[None, :],
                mask=(offs_m[:, None] < ROWS) & time_mask[None, :],
                other=0.0,
            )
            a = a_state + a_edge + a_time
            b = tl.load(
                weight + offs_n[None, :] * total_dim + k[:, None],
                mask=(offs_n[None, :] < OUT_DIM) & (k[:, None] < total_dim),
                other=0.0,
            )
            acc += tl.dot(a, b, input_precision="tf32")

        acc += tl.load(bias + offs_n, mask=offs_n < OUT_DIM, other=0.0)[None, :]
        tl.store(
            out + offs_m[:, None] * OUT_DIM + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < OUT_DIM),
        )


def triton_edge_projection(
    state,
    edge,
    time,
    weight,
    bias,
    edge_src,
    edge_to_eid,
    block_m=32,
    block_n=64,
    block_k=32,
):
    out = torch.empty((time.shape[0], weight.shape[0]), device=time.device, dtype=time.dtype)
    grid = (triton.cdiv(time.shape[0], block_m), triton.cdiv(weight.shape[0], block_n))
    _triton_edge_projection_kernel[grid](
        state,
        edge,
        time,
        weight,
        bias,
        edge_src,
        edge_to_eid,
        out,
        time.shape[0],
        state.shape[1],
        edge.shape[1],
        time.shape[1],
        weight.shape[0],
        block_m,
        block_n,
        block_k,
    )
    return out


def timed_ms(fn, repeat):
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=65_536)
    parser.add_argument("--state-nodes", type=int, default=16_384)
    parser.add_argument("--edges", type=int, default=32_768)
    parser.add_argument("--state-dim", type=int, default=100)
    parser.add_argument("--edge-dim", type=int, default=172)
    parser.add_argument("--time-dim", type=int, default=100)
    parser.add_argument("--out-dim", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6773)
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    import starrygl

    if not hasattr(starrygl, "ops") or not hasattr(
        starrygl.ops, "fused_edge_temporal_linear_forward"
    ):
        raise SystemExit("starrygl fused op is not built")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    state = torch.randn(args.state_nodes, args.state_dim, device=device)
    edge = torch.randn(args.edges, args.edge_dim, device=device)
    time = torch.randn(args.rows, args.time_dim, device=device)
    edge_src = torch.randint(args.state_nodes, (args.rows,), device=device)
    edge_to_eid = torch.randint(args.edges, (args.rows,), device=device)
    linear = torch.nn.Linear(
        args.state_dim + args.edge_dim + args.time_dim,
        args.out_dim,
        device=device,
    )

    def baseline():
        return linear(torch.cat([state[edge_src], edge[edge_to_eid], time], dim=1))

    def fused():
        return starrygl.ops.fused_edge_temporal_linear_forward(
            state.contiguous(),
            edge.contiguous(),
            time.contiguous(),
            linear.weight.contiguous(),
            linear.bias.contiguous(),
            edge_src.contiguous(),
            edge_to_eid.contiguous(),
        )

    def triton_fused():
        return triton_edge_projection(
            state.contiguous(),
            edge.contiguous(),
            time.contiguous(),
            linear.weight.contiguous(),
            linear.bias.contiguous(),
            edge_src.contiguous(),
            edge_to_eid.contiguous(),
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
        )

    with torch.inference_mode():
        torch.testing.assert_close(fused()[:1024], baseline()[:1024], rtol=1e-4, atol=1e-4)
        use_triton = (not args.no_triton) and triton is not None
        if use_triton:
            torch.testing.assert_close(
                triton_fused()[:1024], baseline()[:1024], rtol=1e-2, atol=1e-2
            )
        for _ in range(args.warmup):
            baseline()
            fused()
            if use_triton:
                triton_fused()
        torch.cuda.synchronize()
        baseline_ms = timed_ms(baseline, args.repeat)
        fused_ms = timed_ms(fused, args.repeat)
        triton_ms = timed_ms(triton_fused, args.repeat) if use_triton else []

    baseline_median = statistics.median(baseline_ms)
    fused_median = statistics.median(fused_ms)
    triton_median = statistics.median(triton_ms) if triton_ms else None
    result = {
        "rows": args.rows,
        "state_nodes": args.state_nodes,
        "edges": args.edges,
        "state_dim": args.state_dim,
        "edge_dim": args.edge_dim,
        "time_dim": args.time_dim,
        "out_dim": args.out_dim,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "baseline_median_ms": baseline_median,
        "naive_cuda_fused_median_ms": fused_median,
        "naive_cuda_speedup": baseline_median / fused_median,
        "naive_cuda_reduction_pct": (baseline_median - fused_median)
        / baseline_median
        * 100.0,
    }
    if triton_median is not None:
        result.update(
            {
                "triton_fused_median_ms": triton_median,
                "triton_speedup": baseline_median / triton_median,
                "triton_reduction_pct": (baseline_median - triton_median)
                / baseline_median
                * 100.0,
            }
        )
    print(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
