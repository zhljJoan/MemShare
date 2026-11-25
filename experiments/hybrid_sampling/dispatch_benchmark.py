"""Compare CPU submission of identical LASTFM queries across three launch paths."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from benchmark import replay
from probe import FixedWorkspace, SplitSampler, _fixed_sample


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("partition", type=Path)
    args = parser.parse_args()
    graph, freq, batches = replay(args.csv, args.partition)
    degree = np.diff(graph[0])
    candidates = np.flatnonzero(freq)
    ranked = candidates[
        np.argsort(-freq[candidates] / np.maximum(degree[candidates], 1), kind="stable")
    ]
    nodes = ranked[np.cumsum(degree[ranked]) <= 0.3 * degree.sum()]
    refs = []
    report = {}
    for mode in ("hybrid", "fixed", "graph"):
        start = perf_counter_ns()
        workspace = (
            None if mode == "hybrid" else FixedWorkspace(20, True, mode == "graph")
        )
        precompile_ms = (perf_counter_ns() - start) / 1e6
        sampler = SplitSampler(*graph, nodes, workspace=workspace)
        for j, (root, ts) in enumerate(batches):
            result, _ = sampler.sample(root, ts, batch=j)
            if workspace is None:
                refs.append(result.copy())
            else:
                np.testing.assert_array_equal(result, refs[j])
        samples = []
        for j in range(126):
            start = perf_counter_ns()
            _, stats = sampler.sample(*batches[j % len(batches)], batch=j, profile=True)
            wall = (perf_counter_ns() - start) / 1000
            if j >= 21:
                samples.append((wall, stats["host_submit_us"], stats["dispatch_us"]))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for j in range(12):
                sampler.sample(*batches[j], batch=j)
        gpu = [
            e.time_range.elapsed_us()
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and "_sample" in e.name
        ]
        assert len(gpu) == 12, (mode, len(gpu))
        calls = {
            name: sum(e.name == name for e in prof.events()) / 12
            for name in (
                "cudaLaunchKernel",
                "cuLaunchKernel",
                "cudaGraphLaunch",
                "cudaMemcpyAsync",
            )
        }
        report[mode] = dict(
            zip(
                ("sample_wall_us", "host_submit_us", "dispatch_us"),
                np.median(samples, axis=0),
            )
        )
        report[mode].update(
            precompile_and_load_ms=precompile_ms if workspace else None,
            gpu_kernel_us=float(np.median(gpu)),
            traced_host_calls_per_batch=calls,
            matched_batches=len(refs),
        )
    report["fixed_compiled_signatures"] = len(
        _fixed_sample.cache[torch.cuda.current_device()]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
