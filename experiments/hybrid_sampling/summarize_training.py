"""Check rank/epoch alignment and aggregate independent training and trace runs."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", type=Path, help="directory with native1, hybrid1, nc1, hc1, etc."
    )
    root = parser.parse_args().runs
    summary = {}
    demand = None
    for mode in ("native", "hybrid", "nc", "hc"):
        steady, first, peaks = [], [], []
        for run in (1, 2):
            rows = read_rows(root / f"{mode}{run}" / "epochs.jsonl")
            assert [r["epoch"] for r in rows] == [0, 1, 2]
            counts = [(r["remote_nodes"], r["remote_edges"]) for r in rows]
            if demand is None:
                demand = counts
            assert counts == demand
            for r in rows:
                ranks = r["ranks"]
                assert {x["rank"] for x in ranks} == {0, 1, 2, 3}
                assert all(
                    x["batches"] == 76 and math.isfinite(x["loss"]) for x in ranks
                )
                assert r["critical_train_seconds"] == max(
                    x["train_seconds"] for x in ranks
                )
            first.append(rows[0]["critical_train_seconds"])
            steady.append(mean(r["critical_train_seconds"] for r in rows[1:]))
            peaks.append(
                max(x["peak_cuda_bytes"] for r in rows for x in r["ranks"]) / 2**20
            )
        summary[mode] = dict(
            steady_runs_s=steady,
            steady_mean_s=mean(steady),
            first_epoch_s=first,
            peak_rank_MiB=peaks,
        )
    traces = []
    for rank in range(4):
        rows = read_rows(root / "trace" / f"rank-{rank}.jsonl")
        assert [r["batch"] for r in rows] == list(range(5, 30))
        traces.extend(rows)
    reuse = {}
    for key in ("rows", "edges", "root_times"):
        for window in (1, 4):
            field = f"{key}_w{window}"
            eligible = [r for r in traces if field in r]
            reuse[field] = sum(r[field] for r in eligible) / sum(
                r[key] for r in eligible
            )
    counts = [0, 0, 0, 0]
    for rank in range(4):
        records = (root / "verify" / f"edge-{rank}.tsv").read_text().splitlines()
        assert len(records) == 3 * 75
        for line in records:
            wanted, hits, remote, remote_hits = map(int, line.split())
            assert remote_hits <= hits <= wanted and remote_hits <= remote <= wanted
            for i, n in enumerate([wanted, hits, remote, remote_hits]):
                counts[i] += n
    print(
        json.dumps(
            dict(
                timing=summary,
                reuse=reuse,
                edge_requests=counts[0],
                edge_hits=counts[1],
                remote_requests=counts[2],
                remote_hits=counts[3],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
