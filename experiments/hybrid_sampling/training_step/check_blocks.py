"""Compare full ordinary, reverse and identity block layouts against the training builder."""

import argparse
import json
from pathlib import Path

from types import SimpleNamespace
from unittest.mock import patch

import torch
from starrygl.sample import batch_data
from packed_block import install

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", type=Path, help="trace CUDA runtime calls for the largest fixture")
args = parser.parse_args()
counts = []
origin = (batch_data.to_block, batch_data.to_reversed_block)
install()
gpu = torch.device("cuda:0")


class Sample:
    def __init__(self, n, t, e, o, qt):
        self.fields = n, t, e, o, qt[o] - t

    def sample_nodes(self):
        return self.fields[0]

    def sample_nodes_ts(self):
        return self.fields[1]

    def eid(self):
        return self.fields[2]

    def src_index(self):
        return self.fields[3]

    def delta_ts(self):
        return self.fields[4]


with patch("torch.distributed.get_rank", return_value=0):
    torch.manual_seed(229)
    for count in (0, 1, 37, 11000):
        q = torch.randint(0, 40, (120,))
        qt = torch.randint(10, 20, (len(q),))
        n = torch.randint(40, (count,))
        t = torch.randint(0, 9, (count,))
        e = torch.randint(70, (count,))
        o = torch.randint(len(q), (count,))
        sample = Sample(n, t, e, o, qt)
        graph = SimpleNamespace(
            nids_mapper=torch.randperm(40) + (torch.arange(40) % 4 << 50),
            eids_mapper=torch.randperm(70) + (torch.arange(70) % 4 << 50),
        )
        meta = dict(
            seed=q,
            seed_ts=qt,
            dst_pos_index=torch.arange(40, 80, device=gpu),
            dst_neg_index=torch.arange(80, 120, device=gpu),
            train_ratio_pos=0.4,
        )
        for which in ("forward", "reverse", "identity"):
            reverse = which != "forward"
            ident = which == "identity"
            outputs = []
            for funcs in [origin, (batch_data.to_block, batch_data.to_reversed_block)]:
                func = funcs[int(reverse)]

                def construct():
                    return func(
                        graph,
                        None,
                        (None if ident else [sample], dict(meta)),
                        gpu,
                        **({"identity": ident} if reverse else {}),
                    )

                outputs.append(construct())
                if args.profile and count == 11000:
                    torch.cuda.synchronize()
                    iterations = 20
                    with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
                    ]) as trace:
                        for _ in range(iterations):
                            construct()
                        torch.cuda.synchronize()
                    events = [e for e in trace.key_averages()
                              if e.key.startswith(("cudaLaunch", "cudaMemcpy"))]
                    if not any(e.key.startswith("cudaLaunch") for e in events):
                        raise RuntimeError("CUDA runtime launch events unavailable")
                    counts.append(dict(
                        layout=which, packed=len(outputs) == 2, iterations=iterations,
                        calls_per_block={e.key: e.count / iterations for e in events},
                        host_us_per_block={e.key: e.self_cpu_time_total / iterations for e in events},
                    ))
            (a, an, ae), (b, bn, be) = outputs
            assert torch.equal(an, bn) and torch.equal(ae, be)
            x, y = a[1][0][0], b[1][0][0]
            assert (x.num_src_nodes(), x.num_dst_nodes()) == (
                y.num_src_nodes(),
                y.num_dst_nodes(),
            )
            assert all(torch.equal(l, r) for l, r in zip(x.edges(), y.edges()))
            for attrs in ("srcdata", "dstdata", "edata"):
                lhs, rhs = getattr(x, attrs), getattr(y, attrs)
                assert set(lhs) == set(rhs) and all(
                    torch.equal(lhs[k], rhs[k]) for k in lhs
                )
            for k in a[2]:
                if isinstance(a[2][k], torch.Tensor):
                    assert torch.equal(a[2][k], b[2][k])
print("12 ordinary/reverse/identity checks passed")

if args.profile:
    args.profile.write_text(json.dumps(counts, indent=2) + "\n")
