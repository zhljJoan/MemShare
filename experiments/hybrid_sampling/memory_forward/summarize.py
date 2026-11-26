"""Validate paired workload and report steady time and owner-side forwarding."""
import json
import math
from pathlib import Path
from statistics import mean
from sys import argv

root = Path(argv[1])
for dataset, prefix, epochs, batches in (("LASTFM", "", 5, 76), ("WikiTalk", "w", 3, 457)):
    reference = None
    for mode in ("b", "p", "f"):
        times = []
        for repeat in (1, 2):
            rows = [json.loads(line) for line in
                    (root / f"{prefix}{mode}{repeat}.jsonl").read_text().splitlines()]
            assert [row["epoch"] for row in rows] == list(range(epochs))
            demand = [(row["remote_nodes"], row["remote_edges"]) for row in rows]
            if reference is None:
                reference = demand
            assert reference == demand
            for row in rows:
                assert row["dataset"] == dataset and row["model"] == "TGN_large"
                assert sorted(r["rank"] for r in row["ranks"]) == list(range(4))
                assert all(r["batches"] == batches and math.isfinite(r["loss"])
                           for r in row["ranks"])
                assert row["critical_train_seconds"] == max(r["train_seconds"] for r in row["ranks"])
            times.append(mean(r["critical_train_seconds"] for r in rows[1:]))
        print(dataset, mode, *(f"{t:.4f}" for t in times), f"mean={mean(times):.4f}")
    total = hits = 0
    for rank in range(4):
        lines = (root / dataset / f"rank-{rank}.jsonl").read_text().splitlines()
        assert len(lines) == batches - 1
        for line in lines:
            row = json.loads(line)
            assert 0 <= row["hits"] <= row["rows"] and row["memory_width"] == 100
            total += row["rows"]
            hits += row["hits"]
    print(f"{dataset}: {hits}/{total} forwarded ({100*hits/total:.2f}%)")
