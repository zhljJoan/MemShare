"""Verify complete paired epochs before writing results to CSV."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

DATASETS = ("LASTFM", "WikiTalk", "StackOverflow", "GDELT")
MODELS = ("TGN", "JODIE", "APAN")


def read(path, dataset, model, epochs):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["epoch"] for r in rows] == list(range(epochs)), path
    for row in rows:
        ranks = row["ranks"]
        assert row["dataset"] == dataset and row["model"] == model + "_large"
        assert sorted(r["rank"] for r in ranks) == list(range(4))
        assert len({r["batches"] for r in ranks}) == 1 and ranks[0]["batches"] > 0
        assert all(
            math.isfinite(r["loss"]) and r["alpha"] == 0.3 and r["beta"] == 0.1
            for r in ranks
        )
        assert row["critical_train_seconds"] == max(r["train_seconds"] for r in ranks)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            pilot = dataset == "LASTFM" and model == "TGN"
            epochs = 3 if dataset in DATASETS[:2] else 1
            pairs = 2 if pilot else 1
            baseline = []
            packed = []
            for run in range(1, pairs + 1):
                if pilot:
                    paths = [
                        args.raw / "pilot" / f"{mode}{run}" / "epochs.jsonl"
                        for mode in ("base", "packed")
                    ]
                else:
                    paths = [
                        args.raw
                        / "matrix"
                        / f"{dataset}.{model}"
                        / str(mode)
                        / "epochs.jsonl"
                        for mode in (0, 1)
                    ]
                a, b = [read(p, dataset, model, epochs) for p in paths]
                for x, y in zip(a, b):
                    for key in ("remote_nodes", "remote_edges"):
                        assert x[key] == y[key], (dataset, model, key, x[key], y[key])
                    assert [r["batches"] for r in x["ranks"]] == [
                        r["batches"] for r in y["ranks"]
                    ]
                baseline.append([r["critical_train_seconds"] for r in a])
                packed.append([r["critical_train_seconds"] for r in b])
            base = mean(x[0] for x in baseline)
            opt = mean(x[0] for x in packed)
            row = dict(
                dataset=dataset,
                model=model,
                paired_launches=pairs,
                epochs_per_launch=epochs,
                batches_per_rank=a[0]["ranks"][0]["batches"],
                epoch0_base_s=base,
                epoch0_packed_s=opt,
                epoch0_reduction_pct=100 * (1 - opt / base),
            )
            if epochs > 1:
                base = mean(mean(x[1:]) for x in baseline)
                opt = mean(mean(x[1:]) for x in packed)
                row.update(
                    steady_base_s=base,
                    steady_packed_s=opt,
                    steady_reduction_pct=100 * (1 - opt / base),
                )
            records.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
