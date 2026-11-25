"""Replay disjoint, consecutive rank-0 training edges against a static CSR."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from probe import SplitSampler


def replay(csv, partition):
    events = pd.read_csv(csv, usecols=["src", "dst", "time", "ext_roll"])
    src, dst, ts, split = (
        events[k].to_numpy(dtype=np.int64) for k in ("src", "dst", "time", "ext_roll")
    )
    nodes = max(src.max(), dst.max()) + 1
    local_node = np.zeros(nodes, dtype=bool)
    for filename in ["output0.txt", "outputshared.txt"]:
        local_node[np.loadtxt(partition / filename, dtype=np.int64)] = True
    local_edge = np.zeros(len(src), dtype=bool)
    local_edge[np.loadtxt(partition / "edge_output0.txt", dtype=np.int64)] = True
    owner = np.r_[src, dst]
    order = np.lexsort((np.tile(ts, 2), owner))
    nbr = np.r_[dst, src][order]
    ptr = np.r_[0, np.cumsum(np.bincount(owner, minlength=nodes))]
    graph = (
        ptr,
        nbr,
        np.tile(ts, 2)[order],
        ~(np.tile(local_edge, 2)[order] & local_node[nbr]),
    )
    ids = np.flatnonzero(local_edge & (split == 0))
    start = len(ids) // 2
    batches = []
    for idx in range(26):
        block = ids[start + idx * 3000 : start + (idx + 1) * 3000]
        assert len(block) == 3000
        batches.append((np.r_[src[block], dst[block]], np.tile(ts[block], 2)))
    freq = np.bincount(np.concatenate([r for r, _ in batches[:5]]), minlength=nodes)
    return graph, freq, batches[5:]


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
    accum = np.cumsum(degree[ranked])
    expected = []
    report = {}
    for fraction in [0, 0.1, 0.3, 1]:
        nodes = (
            np.arange(len(freq))
            if fraction == 1
            else ranked[accum <= fraction * degree.sum()]
        )
        sampler = SplitSampler(*graph, nodes)
        for i, (roots, times) in enumerate(batches):
            result, _ = sampler.sample(roots, times, batch=i)
            if fraction == 0:
                expected.append(result)
            else:
                np.testing.assert_array_equal(result, expected[i])
        for j in range(25):
            sampler.sample(*batches[j % len(batches)], batch=j)
        samples = []
        for j in range(105):
            cpu_start = time.process_time_ns()
            start = time.perf_counter_ns()
            _, stats = sampler.sample(*batches[j % len(batches)], batch=j)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
            samples.append(
                [elapsed, cpu_ms, stats["gpu_kernel_ms"], stats["gpu_queries"]]
            )
        med = np.median(samples, axis=0)
        report[str(fraction)] = dict(
            wall_ms=med[0],
            cpu_core_ms=med[1],
            gpu_kernel_ms=med[2],
            gpu_queries=med[3],
            cache_MiB=sampler.cache_bytes / 2**20,
            wall_p95_ms=np.percentile(np.array(samples)[:, 0], 95),
        )
        del sampler
    print(
        json.dumps(
            dict(
                dataset=str(args.csv),
                partition=args.partition.name,
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                queries_per_batch=6000,
                matched_batches=len(expected),
                measurements_per_mode=105,
                results=report,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
