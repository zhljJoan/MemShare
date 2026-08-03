#!/usr/bin/env python3
"""Summarize paired isolated coordinated-sampling experiment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


DATASET_EPOCHS = {
    "WIKI": 50,
    "LASTFM": 50,
    "WikiTalk": 10,
    "Flight": 10,
    "GDELT": 10,
    "StackOverflow": 10,
}
LARGE_DATASETS = {"GDELT", "StackOverflow"}
EFFICIENCY_KEYS = (
    "critical_train_seconds",
    "max_rank_batch_p95_seconds",
    "remote_nodes",
    "remote_edges",
    "local_nodes",
    "local_edges",
)
QUALITY_KEYS = ("val_ap", "val_auc", "test_ap", "test_auc")
PHASES = (
    "sampler",
    "all_to_all_route",
    "all_to_all_launch",
    "fetch_wait",
    "node_fetch_wait",
    "edge_fetch_wait",
    "sampler_future_wait",
)
SAMPLE_TYPES = {
    "baseline": "boundery_recent_decay",
    "coordinated": "boundery_recent_decay_coordinated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[6773, 6774, 6775]
    )
    parser.add_argument(
        "--large-seeds",
        nargs="+",
        type=int,
        default=[6773],
        help=(
            "Seeds used by GDELT and StackOverflow. These expensive "
            "supplementary runs default to one preliminary paired seed."
        ),
    )
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percent_change(baseline: float, coordinated: float) -> float:
    return (coordinated / baseline - 1.0) * 100.0


def percentage_point_change(baseline: float, coordinated: float) -> float:
    return (coordinated - baseline) * 100.0


def mean_stdev(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def load_epoch_records(
    root: Path,
    variant: str,
    dataset: str,
    epochs: int,
    seed: int,
    group_size: int,
) -> list[dict]:
    path = root / "metrics" / (
        "{}__{}__{}ep__seed{}__g{}.jsonl".format(
            variant, dataset, epochs, seed, group_size
        )
    )
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(records) != epochs:
        raise ValueError(
            "{} contains {} records, expected {}".format(
                path, len(records), epochs
            )
        )
    expected_epochs = list(range(epochs))
    if [record.get("epoch") for record in records] != expected_epochs:
        raise ValueError("{} has a non-contiguous epoch sequence".format(path))
    for record in records:
        if record.get("world_size") != 8:
            raise ValueError("{} is not an eight-rank run".format(path))
        if record.get("dataset") != dataset:
            raise ValueError("{} has the wrong dataset".format(path))
        if record.get("sample_type") != SAMPLE_TYPES[variant]:
            raise ValueError("{} has the wrong sample type".format(path))
        if any(record["quality"].get(key) is None for key in QUALITY_KEYS):
            raise ValueError("{} is missing quality metrics".format(path))
    return records


def load_launcher(
    root: Path,
    variant: str,
    dataset: str,
    epochs: int,
    seed: int,
    group_size: int,
) -> dict:
    path = root / "launcher" / (
        "{}__{}__{}ep__seed{}__g{}.json".format(
            variant, dataset, epochs, seed, group_size
        )
    )
    launcher = json.loads(path.read_text())
    if launcher.get("local_return_code") != 0:
        raise ValueError("{} has a nonzero local return code".format(path))
    if launcher.get("remote_return_code") != 0:
        raise ValueError("{} has a nonzero remote return code".format(path))
    network = launcher.get("network", {})
    if not network.get("socket_interface_seen"):
        raise ValueError("{} did not record the Socket interface".format(path))
    if not network.get("socket_network_selected"):
        raise ValueError("{} did not select NCCL Socket".format(path))
    if network.get("ib_transport_seen"):
        raise ValueError("{} used an IB transport".format(path))
    return launcher


def first_target_epoch(records: list[dict], target: float) -> int | None:
    for record in records:
        if record["quality"]["val_ap"] >= target:
            return int(record["epoch"])
    return None


def cumulative_train_seconds(records: list[dict], epoch: int) -> float:
    return sum(
        float(record["critical_train_seconds"])
        for record in records
        if int(record["epoch"]) <= epoch
    )


def summarize_dataset(
    root: Path,
    dataset: str,
    epochs: int,
    seeds: list[int],
    group_size: int,
) -> dict:
    paired = []
    for seed in seeds:
        baseline = load_epoch_records(
            root, "baseline", dataset, epochs, seed, group_size
        )
        coordinated = load_epoch_records(
            root, "coordinated", dataset, epochs, seed, group_size
        )
        baseline_launcher = load_launcher(
            root, "baseline", dataset, epochs, seed, group_size
        )
        coordinated_launcher = load_launcher(
            root, "coordinated", dataset, epochs, seed, group_size
        )
        measured_baseline = baseline[1:] if len(baseline) > 1 else baseline
        measured_coordinated = (
            coordinated[1:] if len(coordinated) > 1 else coordinated
        )
        efficiency = {}
        for key in EFFICIENCY_KEYS:
            baseline_mean = statistics.mean(
                float(record[key]) for record in measured_baseline
            )
            coordinated_mean = statistics.mean(
                float(record[key]) for record in measured_coordinated
            )
            efficiency[key] = percent_change(
                baseline_mean, coordinated_mean
            )

        phase_change = {}
        for phase in PHASES:
            baseline_mean = statistics.mean(
                float(
                    record["phase_timing"][phase][
                        "critical_rank_seconds"
                    ]
                )
                for record in measured_baseline
            )
            coordinated_mean = statistics.mean(
                float(
                    record["phase_timing"][phase][
                        "critical_rank_seconds"
                    ]
                )
                for record in measured_coordinated
            )
            phase_change[phase] = percent_change(
                baseline_mean, coordinated_mean
            )

        final_quality = {
            key: percentage_point_change(
                baseline[-1]["quality"][key],
                coordinated[-1]["quality"][key],
            )
            for key in QUALITY_KEYS
        }
        best_quality = {
            key: percentage_point_change(
                max(record["quality"][key] for record in baseline),
                max(record["quality"][key] for record in coordinated),
            )
            for key in QUALITY_KEYS
        }

        target = 0.99 * max(
            record["quality"]["val_ap"] for record in baseline
        )
        baseline_target_epoch = first_target_epoch(baseline, target)
        coordinated_target_epoch = first_target_epoch(coordinated, target)
        target_record = {
            "target_val_ap": target,
            "baseline_epoch": baseline_target_epoch,
            "coordinated_epoch": coordinated_target_epoch,
            "epoch_delta": None,
            "train_seconds_change_percent": None,
        }
        if (
            baseline_target_epoch is not None
            and coordinated_target_epoch is not None
        ):
            target_record["epoch_delta"] = (
                coordinated_target_epoch - baseline_target_epoch
            )
            target_record["train_seconds_change_percent"] = percent_change(
                cumulative_train_seconds(baseline, baseline_target_epoch),
                cumulative_train_seconds(
                    coordinated, coordinated_target_epoch
                ),
            )

        paired.append(
            {
                "seed": seed,
                "efficiency_change_percent": efficiency,
                "phase_change_percent": phase_change,
                "final_quality_change_percentage_points": final_quality,
                "best_quality_change_percentage_points": best_quality,
                "wall_change_percent": percent_change(
                    baseline_launcher["wall_seconds"],
                    coordinated_launcher["wall_seconds"],
                ),
                "wire_bytes_change_percent": percent_change(
                    baseline_launcher["nic_bytes"]["estimated_wire_bytes"],
                    coordinated_launcher["nic_bytes"]["estimated_wire_bytes"],
                ),
                "time_to_99pct_baseline_best_val_ap": target_record,
            }
        )

    aggregate = {
        "efficiency_change_percent": {
            key: mean_stdev(
                [item["efficiency_change_percent"][key] for item in paired]
            )
            for key in EFFICIENCY_KEYS
        },
        "phase_change_percent": {
            phase: mean_stdev(
                [item["phase_change_percent"][phase] for item in paired]
            )
            for phase in PHASES
        },
        "final_quality_change_percentage_points": {
            key: mean_stdev(
                [
                    item["final_quality_change_percentage_points"][key]
                    for item in paired
                ]
            )
            for key in QUALITY_KEYS
        },
        "best_quality_change_percentage_points": {
            key: mean_stdev(
                [
                    item["best_quality_change_percentage_points"][key]
                    for item in paired
                ]
            )
            for key in QUALITY_KEYS
        },
        "wall_change_percent": mean_stdev(
            [item["wall_change_percent"] for item in paired]
        ),
        "wire_bytes_change_percent": mean_stdev(
            [item["wire_bytes_change_percent"] for item in paired]
        ),
        "time_to_99pct_baseline_best_val_ap": {
            "epoch_delta": mean_stdev(
                [
                    item["time_to_99pct_baseline_best_val_ap"][
                        "epoch_delta"
                    ]
                    for item in paired
                    if item["time_to_99pct_baseline_best_val_ap"][
                        "epoch_delta"
                    ]
                    is not None
                ]
            ),
            "train_seconds_change_percent": mean_stdev(
                [
                    item["time_to_99pct_baseline_best_val_ap"][
                        "train_seconds_change_percent"
                    ]
                    for item in paired
                    if item["time_to_99pct_baseline_best_val_ap"][
                        "train_seconds_change_percent"
                    ]
                    is not None
                ]
            ),
        },
    }
    return {
        "dataset": dataset,
        "epochs": epochs,
        "seeds": seeds,
        "group_size": group_size,
        "aggregate": aggregate,
        "paired_runs": paired,
    }


def main() -> int:
    args = parse_args()
    datasets = {
        dataset: summarize_dataset(
            args.run_root,
            dataset,
            epochs,
            args.large_seeds if dataset in LARGE_DATASETS else args.seeds,
            args.group_size,
        )
        for dataset, epochs in DATASET_EPOCHS.items()
    }
    summary = {
        "run_root": str(args.run_root),
        "datasets": datasets,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
