import argparse
import json
import pathlib
import statistics
import sys

import dgl
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from starrygl.module.compact_gru import compact_gru_cell
from starrygl.module import triton_projected_segment_attention, triton_segment_attention


def _measure(fn, repeat):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _run_benchmark(baseline, fused, warmup, repeat, rounds):
    for _ in range(warmup):
        baseline()
        fused()
    baseline_ms = []
    fused_ms = []
    for iteration in range(rounds):
        order = ((baseline, baseline_ms), (fused, fused_ms))
        if iteration % 2:
            order = reversed(order)
        for fn, results in order:
            results.append(_measure(fn, repeat))
    baseline_median = statistics.median(baseline_ms)
    fused_median = statistics.median(fused_ms)
    return {
        "baseline_median_ms": baseline_median,
        "fused_median_ms": fused_median,
        "fused_reduction_pct": (baseline_median - fused_median) / baseline_median * 100.0,
        "speedup": baseline_median / fused_median,
        "baseline_rounds_ms": baseline_ms,
        "fused_rounds_ms": fused_ms,
    }


def _profile_once(step, warmup):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    step()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def _segment(args):
    destinations, degree, heads, head_dim = 6000, 18, 2, 50
    edges = destinations * degree
    device = torch.device("cuda")
    q = torch.randn(destinations, heads, head_dim, device=device, requires_grad=True)
    k = torch.randn(edges, heads, head_dim, device=device, requires_grad=True)
    v = torch.randn(edges, heads, head_dim, device=device, requires_grad=True)
    dst_ptr = torch.arange(destinations + 1, device=device) * degree
    segment_dst = torch.arange(destinations, device=device).repeat_interleave(degree)
    edge_ids = (torch.arange(edges, device=device)
                if args.segment_edge_order == "identity"
                else torch.randperm(edges, device=device))
    edge_dst = torch.empty_like(segment_dst)
    edge_dst[edge_ids] = segment_dst
    block = dgl.create_block(
        (torch.zeros_like(edge_dst), edge_dst),
        num_src_nodes=1, num_dst_nodes=destinations, device=device)
    _, _, block_edge_ids = block.adj_tensors("csc")
    dropout_mask = torch.rand(edges, heads, device=device) >= 0.2
    grad = torch.randn_like(q)

    kwargs = dict(dropout_p=0.2, training=True, dropout_mask=dropout_mask,
                  edge_ids=block_edge_ids)

    def baseline_output():
        score = F.leaky_relu(
            (q.index_select(0, edge_dst) * k).sum(-1), negative_slope=0.2)
        weight = dgl.ops.edge_softmax(block, score)
        weight = weight * dropout_mask / 0.8
        block.edata["v"] = (weight[:, :, None] * v).flatten(1)
        block.update_all(dgl.function.copy_e("v", "m"), dgl.function.sum("m", "h"))
        return block.dstdata["h"].reshape(destinations, heads, head_dim)

    def fused_output():
        return triton_segment_attention.attention(
            q, k, v, dst_ptr, validate=False, **kwargs)

    def baseline():
        return torch.autograd.grad(baseline_output(), (q, k, v), grad)

    def fused():
        return torch.autograd.grad(fused_output(), (q, k, v), grad)

    torch.testing.assert_close(
        fused_output(), baseline_output(), rtol=5e-4, atol=5e-4)
    expected = baseline()
    actual = fused()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=5e-4, atol=5e-4)
    if args.profile:
        _profile_once(fused, args.warmup)
        return {"kernel": "segment_attention", "profile_only": True}
    result = _run_benchmark(baseline, fused, args.warmup, args.repeat, args.rounds)
    result.update({
        "kernel": "segment_attention",
        "shape": {"destinations": destinations, "edges": edges,
                  "heads": heads, "head_dim": head_dim, "degree": degree,
                  "edge_order": args.segment_edge_order},
    })
    return result


def _gru(args):
    states, objects, mail_dim, time_dim, hidden_dim = 13865, 1579, 372, 100, 100
    device = torch.device("cuda")
    state_to_uid = torch.randint(objects, (states,), device=device)
    mailbox = torch.randn(objects, mail_dim, device=device)
    memory = torch.randn(objects, hidden_dim, device=device)
    state_memory = memory.index_select(0, state_to_uid)
    time_feat = torch.randn(states, time_dim, device=device, requires_grad=True)
    cell = torch.nn.GRUCell(mail_dim + time_dim, hidden_dim, device=device)
    grad = torch.randn(states, hidden_dim, device=device)
    parameters = (time_feat, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)

    def baseline_output():
        input_gates = F.linear(mailbox, cell.weight_ih[:, :mail_dim], None)
        input_gates = input_gates.index_select(0, state_to_uid)
        input_gates = torch.addmm(
            input_gates, time_feat, cell.weight_ih[:, mail_dim:].t())
        hidden_gates = F.linear(memory, cell.weight_hh, None)
        hidden_gates = hidden_gates.index_select(0, state_to_uid)
        return torch.ops.aten._thnn_fused_gru_cell(
            input_gates, hidden_gates, state_memory,
            cell.bias_ih, cell.bias_hh)[0]

    def fused_output():
        return compact_gru_cell(
            cell, mailbox, memory, time_feat, state_to_uid, state_memory)

    def baseline():
        return torch.autograd.grad(baseline_output(), parameters, grad)

    def fused():
        return torch.autograd.grad(fused_output(), parameters, grad)

    torch.testing.assert_close(
        fused_output(), baseline_output(), rtol=2e-4, atol=2e-4)
    expected = baseline()
    actual = fused()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
    if args.profile:
        _profile_once(fused, args.warmup)
        return {"kernel": "compact_gru", "profile_only": True}
    result = _run_benchmark(baseline, fused, args.warmup, args.repeat, args.rounds)
    result.update({
        "kernel": "compact_gru",
        "shape": {"states": states, "objects": objects, "mail_dim": mail_dim,
                  "time_dim": time_dim, "hidden_dim": hidden_dim},
    })
    return result


def _projected(args):
    destinations, degree, heads, head_dim = 4096, 16, 2, 50
    edges, states, edge_objects = destinations * degree, 8192, 4096
    width = 2 * heads * head_dim
    device = torch.device("cuda")
    q = torch.randn(destinations, heads, head_dim, device=device, requires_grad=True)
    state = torch.randn(states, width, device=device, requires_grad=True)
    edge = torch.randn(edge_objects, width, device=device, requires_grad=True)
    time = torch.randn(edges, width, device=device, requires_grad=True)
    edge_src = torch.randint(states, (edges,), device=device)
    edge_obj = torch.randint(edge_objects, (edges,), device=device)
    dst_ptr = torch.arange(destinations + 1, device=device) * degree
    runs = torch.arange(edges, device=device).reshape(-1, 64)
    edge_ids = runs[torch.randperm(runs.shape[0], device=device)].reshape(-1)
    dropout_mask = torch.rand(edges, heads, device=device) >= 0.2
    grad = torch.randn_like(q)

    def preordered_output():
        return triton_projected_segment_attention.attention(
            q, state, edge, time.index_select(0, edge_ids),
            edge_src.index_select(0, edge_ids), dst_ptr,
            dropout_p=0.2, training=True,
            dropout_mask=dropout_mask.index_select(0, edge_ids),
            edge_obj=edge_obj.index_select(0, edge_ids), validate=False,
        )

    def direct_output():
        return triton_projected_segment_attention.attention(
            q, state, edge, time, edge_src, dst_ptr,
            dropout_p=0.2, training=True, dropout_mask=dropout_mask,
            edge_obj=edge_obj, edge_ids=edge_ids, validate=False,
        )

    inputs = (q, state, edge, time)

    def preordered():
        return torch.autograd.grad(preordered_output(), inputs, grad)

    def direct():
        return torch.autograd.grad(direct_output(), inputs, grad)

    torch.testing.assert_close(direct_output(), preordered_output(), rtol=2e-5, atol=2e-5)
    for got, want in zip(direct(), preordered()):
        torch.testing.assert_close(got, want, rtol=5e-4, atol=5e-4)
    result = _run_benchmark(preordered, direct, args.warmup, args.repeat, args.rounds)
    result.update({
        "kernel": "projected_segment_attention_edge_ids",
        "baseline_variant": "preordered_index_select",
        "fused_variant": "direct_edge_ids",
        "shape": {"destinations": destinations, "edges": edges,
                  "states": states, "edge_objects": edge_objects,
                  "heads": heads, "head_dim": head_dim, "degree": degree,
                  "edge_id_adjacent_fraction": torch.mean(
                      (edge_ids[1:] == edge_ids[:-1] + 1).float()).item()},
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", choices=("segment", "gru", "projected"))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6773)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--segment-edge-order", choices=("random", "identity"),
                        default="random")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)
    if args.kernel == "segment":
        result = _segment(args)
    elif args.kernel == "gru":
        result = _gru(args)
    else:
        result = _projected(args)
    result.update({
        "gpu": torch.cuda.get_device_name(),
        "dtype": "float32",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rounds": args.rounds,
        "correctness": True,
    })
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
