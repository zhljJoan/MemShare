"""Experimental adapter for the single-layer training path with ordinary blocks."""

import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from probe import SplitSampler
from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler


class Block:
    def __init__(self, rows, positions, graph, eid, times):
        self.rows = torch.from_numpy(rows)
        self.edges = torch.from_numpy(eid[positions])
        self.nodes = torch.from_numpy(graph[1][positions])
        ts = graph[2][positions]
        self.ts = torch.from_numpy(ts)
        self.delta = torch.from_numpy(times[rows] - ts)

    def src_index(self):
        return self.rows

    def eid(self):
        return self.edges

    def sample_nodes(self):
        return self.nodes

    def sample_nodes_ts(self):
        return self.ts

    def delta_ts(self):
        return self.delta


def install():
    init = NeighborSampler.__init__
    native = NeighborSampler.sample_from_nodes

    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if self.graph_name != "train":
            return
        assert self.num_layers == 1 and not self.hierarchical_compaction
        assert self.policy == "boundery_recent_decay_coordinated"
        self.batch_number = 0
        self.calibration = []
        self.policy_args = kwargs
        self.history = deque(maxlen=4)

    def sample(self, nodes, ts, **kwargs):
        if self.graph_name != "train":
            return native(self, nodes, ts, **kwargs)
        batch = self.batch_number
        self.batch_number += 1
        roots = nodes.numpy()
        if batch < 5:
            self.calibration.append(roots.copy())
            return native(self, nodes, ts, **kwargs)
        options = self.policy_args
        if batch == 5:
            start = time.perf_counter()
            data = self.graph_data
            # Match native tie ordering before grouping rows.
            stamp, order = data.edge_ts.sort()
            target = data.edge_index[1, order].numpy()
            by_row = np.argsort(target, kind="stable")
            ptr = np.r_[0, np.cumsum(np.bincount(target, minlength=self.num_nodes))]
            nbr = data.edge_index[0, order].numpy()[by_row]
            self.eids = data.eid[order].numpy()[by_row]
            rank = options["local_part"]
            remote = (options["edge_part"].numpy()[self.eids] != rank) | (
                options["node_part"].numpy()[nbr] != rank
            )
            freq = np.bincount(
                np.concatenate(self.calibration), minlength=self.num_nodes
            )
            degree = np.diff(ptr)
            candidates = np.flatnonzero(freq)
            ranked = candidates[
                np.argsort(
                    -freq[candidates] / np.maximum(degree[candidates], 1), kind="stable"
                )
            ]
            cached = ranked[np.cumsum(degree[ranked]) <= 0.3 * len(nbr)]
            self.split = SplitSampler(
                ptr,
                nbr,
                stamp.numpy()[by_row],
                remote,
                cached,
                fanout=self.fanout[0],
                threads=self.workers,
            )
            print(
                f"split_setup rank={rank} seconds={time.perf_counter() - start:.3f} bytes={self.split.cache_bytes}",
                flush=True,
            )
        result, _ = self.split.sample(
            roots,
            ts.numpy(),
            batch=batch,
            beta=self.ada_param.beta,
            group=options["coordinated_group_size"],
            seed=options["coordinated_seed"],
        )
        rows, cols = np.where(result >= 0)
        block = Block(rows, result[rows, cols], self.split.graph, self.eids, ts.numpy())
        if os.getenv("MEMSHARE_REUSE_TRACE") and batch < 30:
            reference, _ = native(self, nodes, ts, **kwargs)
            for name in (
                "src_index",
                "eid",
                "sample_nodes",
                "sample_nodes_ts",
                "delta_ts",
            ):
                assert torch.equal(
                    getattr(reference[0], name)(), getattr(block, name)()
                ), name
            keys = [
                np.unique(roots),
                np.unique(block.edges.numpy()),
                np.unique(np.rec.fromarrays([roots, ts.numpy()], names="node,ts")),
            ]
            row = dict(batch=batch, rank=options["local_part"], queries=len(roots))
            for i, label in enumerate(["rows", "edges", "root_times"]):
                row[label] = len(keys[i])
                for window in (1, 4):
                    prev = list(self.history)[-window:]
                    if len(prev) == window:
                        old = np.unique(np.concatenate([p[i] for p in prev]))
                        row[f"{label}_w{window}"] = len(
                            np.intersect1d(keys[i], old, assume_unique=True)
                        )
            self.history.append(keys)
            with (
                Path(os.environ["MEMSHARE_REUSE_TRACE"]) / f"rank-{row['rank']}.jsonl"
            ).open("a") as f:
                print(json.dumps(row), file=f)
        return [block], {}

    NeighborSampler.__init__ = initialize
    NeighborSampler.sample_from_nodes = sample
