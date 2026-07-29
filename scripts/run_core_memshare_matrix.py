#!/usr/bin/env python3
"""Run a CoRe-MemShare screening variant on all four required datasets.

The dataset matrix is deliberately fixed.  A variant is never marked complete
unless WIKI, LASTFM, WikiTalk, and Flight all produced structured metrics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Dict, Iterable, List


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY / "examples"
DATASETS = ("WIKI", "LASTFM", "WikiTalk", "Flight")
SPEED_PARTITION = Path("/home/zlj/SPEED/partition/divided_nodes_seed_starrygl")
FLIGHT_PARTITION = REPOSITORY / "divided_nodes_seed_starrygl"

COMMON_ARGUMENTS = (
    "--model",
    "TGN_large",
    "--topk",
    "0.1",
    "--probability",
    "0.1",
    "--memory_type",
    "historical",
    "--shared_memory_ssim",
    "0.3",
    "--core_min_probability",
    "0.02",
    "--core_group_size",
    "8",
    "--core_neg_hardness_beta",
    "1.0",
    "--core_neg_exploration",
    "0.1",
    "--core_freeze_adaptive_parameters",
)

VARIANTS = {
    "baseline": (
        "--sample_type",
        "boundery_recent_decay",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
    ),
    "residual": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
    ),
    "core-static": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
        "--core_disable_memory_predictor",
    ),
    "coordinated-decay": (
        "--sample_type",
        "boundery_recent_decay_coordinated",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
    ),
    "core-independent": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
        "--core_disable_memory_predictor",
        "--core_group_size",
        "1",
    ),
    "core-clipped": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
        "--core_disable_memory_predictor",
        "--core_attention_probability_floor",
        "0.1",
    ),
    "core-uncorrected": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
        "--core_disable_memory_predictor",
        "--core_attention_probability_floor",
        "1",
    ),
    "core-p15": (
        "--sample_type",
        "boundery_recent_core",
        "--probability",
        "0.15",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
        "--core_disable_memory_predictor",
    ),
    "core-p20": (
        "--sample_type",
        "boundery_recent_core",
        "--probability",
        "0.2",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
        "--core_disable_memory_predictor",
    ),
    "core-top1": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "1",
        "--core_disable_memory_predictor",
        "--core_deterministic_recent",
        "1",
    ),
    "edge-cache": (
        "--sample_type",
        "boundery_recent_decay",
        "--core_edge_cache_ratio",
        "0.01",
        "--core_neg_pool_factor",
        "1",
    ),
    "edge-cache-frozen": (
        "--sample_type",
        "boundery_recent_decay",
        "--core_edge_cache_ratio",
        "0.01",
        "--core_cache_max_replacements",
        "0",
        "--core_cache_simple_admission",
        "--core_neg_pool_factor",
        "1",
    ),
    "residual-cache": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0.01",
        "--core_neg_pool_factor",
        "1",
    ),
    "hard-negative": (
        "--sample_type",
        "boundery_recent_decay",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "4",
    ),
    "cost-negative": (
        "--sample_type",
        "boundery_recent_decay",
        "--core_edge_cache_ratio",
        "0",
        "--core_neg_pool_factor",
        "2",
        "--core_neg_hardness_beta",
        "0",
        "--core_neg_embedding_capacity",
        "0",
    ),
    "full": (
        "--sample_type",
        "boundery_recent_core",
        "--core_edge_cache_ratio",
        "0.01",
        "--core_neg_pool_factor",
        "4",
    ),
}


def _parse_args() -> argparse.Namespace:
    default_python = Path(
        "/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run each selected variant on the mandatory "
            "WIKI/LASTFM/WikiTalk/Flight matrix."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument(
        "--screen-epochs",
        type=int,
        default=0,
        help='run full train/validation/test epochs instead of batch-only smoke',
    )
    parser.add_argument(
        "--dataset-epochs",
        nargs="+",
        default=(),
        metavar="DATASET=EPOCHS",
        help=(
            "override --screen-epochs per dataset; when supplied, all four "
            "fixed datasets must be listed"
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6773)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python,
    )
    parser.add_argument(
        "--native-lib",
        type=Path,
        default=Path(
            "/tmp/core_memshare_script_build/libstarrygl_sampler.so"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "outputs" / "core_memshare_screen",
    )
    parser.add_argument(
        "--run-name",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="")
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("--batches must be positive")
    if args.screen_epochs < 0:
        parser.error("--screen-epochs cannot be negative")
    dataset_epochs: Dict[str, int] = {}
    for assignment in args.dataset_epochs:
        try:
            dataset, raw_epochs = assignment.split("=", 1)
            epochs = int(raw_epochs)
        except ValueError:
            parser.error(
                "--dataset-epochs entries must use DATASET=EPOCHS"
            )
        if dataset not in DATASETS:
            parser.error(
                "unknown dataset in --dataset-epochs: {}".format(dataset)
            )
        if dataset in dataset_epochs:
            parser.error(
                "duplicate dataset in --dataset-epochs: {}".format(dataset)
            )
        if epochs <= 0:
            parser.error("--dataset-epochs values must be positive")
        dataset_epochs[dataset] = epochs
    if dataset_epochs and set(dataset_epochs) != set(DATASETS):
        missing_datasets = sorted(set(DATASETS) - set(dataset_epochs))
        parser.error(
            "--dataset-epochs must cover all four datasets; missing {}".format(
                ", ".join(missing_datasets)
            )
        )
    if dataset_epochs and args.screen_epochs:
        parser.error(
            "--dataset-epochs and --screen-epochs are mutually exclusive"
        )
    args.dataset_epochs = dataset_epochs
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.nproc_per_node != 4:
        parser.error(
            "the checked-in partitions are 4-way; --nproc-per-node must be 4"
        )
    return args


def _partition_root(dataset: str) -> Path:
    return FLIGHT_PARTITION if dataset == "Flight" else SPEED_PARTITION


def _partition_directory(dataset: str) -> Path:
    return (
        _partition_root(dataset)
        / dataset
        / "123457"
        / f"{dataset}_4parts_top0.1"
    )


def _validate(args: argparse.Namespace) -> None:
    missing = []
    if not args.python.is_file():
        missing.append(str(args.python))
    if not args.native_lib.is_file():
        missing.append(str(args.native_lib))
    if not (EXAMPLES / "train_boundery.py").is_file():
        missing.append(str(EXAMPLES / "train_boundery.py"))
    for dataset in DATASETS:
        directory = _partition_directory(dataset)
        required = [
            directory / "outputshared.txt",
            *[directory / f"output{rank}.txt" for rank in range(4)],
            *[directory / f"edge_output{rank}.txt" for rank in range(4)],
        ]
        missing.extend(str(path) for path in required if not path.is_file())
    if missing:
        raise SystemExit(
            "required matrix inputs are missing:\n  " + "\n  ".join(missing)
        )


def _valid_result(path: Path, *, dataset: str, variant: str) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        result.get("dataset") == dataset
        and result.get("experiment") == variant
        and result.get("result", {}).get("status") == "completed"
        and result.get("result", {}).get("loss_is_finite") is True
    )


def _command(
    args: argparse.Namespace,
    *,
    dataset: str,
    variant: str,
    repeat: int,
    metrics_path: Path,
) -> List[str]:
    torchrun = args.python.parent / "torchrun"
    seed = args.seed + repeat
    epoch_count = args.dataset_epochs.get(dataset, args.screen_epochs)
    run_limit = (
        ["--core_screen_epochs", str(epoch_count)]
        if epoch_count > 0
        else ["--core_smoke_batches", str(args.batches)]
    )
    return [
        str(torchrun),
        "--standalone",
        f"--nproc-per-node={args.nproc_per_node}",
        "train_boundery.py",
        "--dataname",
        dataset,
        "--partition",
        str(_partition_root(dataset)),
        "--seed",
        str(seed),
        *run_limit,
        "--core_metrics_path",
        str(metrics_path),
        "--core_experiment_name",
        variant,
        *COMMON_ARGUMENTS,
        *VARIANTS[variant],
    ]


def _tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(content[-lines:])


def _run_one(
    args: argparse.Namespace,
    *,
    dataset: str,
    variant: str,
    repeat: int,
    run_directory: Path,
) -> Dict[str, Any]:
    stem = f"{variant}__{dataset}__r{repeat}"
    metrics_path = run_directory / "metrics" / f"{stem}.json"
    log_path = run_directory / "logs" / f"{stem}.log"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.force and _valid_result(
        metrics_path, dataset=dataset, variant=variant
    ):
        print(f"[resume] {variant:13s} {dataset:8s} repeat={repeat}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics_path.unlink(missing_ok=True)
    command = _command(
        args,
        dataset=dataset,
        variant=variant,
        repeat=repeat,
        metrics_path=metrics_path,
    )
    environment = os.environ.copy()
    environment["STARRYGL_SAMPLER_LIBRARY"] = str(args.native_lib)
    environment["OMP_NUM_THREADS"] = "10"
    environment["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"[start ] {variant:13s} {dataset:8s} repeat={repeat}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=EXAMPLES,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = started + args.timeout_minutes * 60.0
        next_heartbeat = started + 30.0
        timed_out = False
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            if now >= next_heartbeat:
                print(
                    f"[wait  ] {variant:13s} {dataset:8s} "
                    f"elapsed={now - started:.0f}s"
                )
                next_heartbeat = now + 30.0
            time.sleep(2.0)
        return_code = process.returncode

    elapsed = time.monotonic() - started
    if (
        not timed_out
        and return_code == 0
        and _valid_result(metrics_path, dataset=dataset, variant=variant)
    ):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        report["runner"] = {
            "elapsed_seconds": elapsed,
            "log": str(log_path),
            "return_code": return_code,
        }
        metrics_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        step_seconds = _nested(
            report,
            "run_summary",
            "mean_seconds_per_batch",
            default=_nested(
                report, "result", "mean_seconds_per_batch"
            ),
        )
        print(
            f"[done  ] {variant:13s} {dataset:8s} "
            f"elapsed={elapsed:.1f}s "
            f"step={step_seconds:.4f}s"
        )
        return report

    failure = {
        "schema_version": 1,
        "experiment": variant,
        "dataset": dataset,
        "result": {
            "status": "timeout" if timed_out else "failed",
            "loss_is_finite": False,
        },
        "runner": {
            "elapsed_seconds": elapsed,
            "log": str(log_path),
            "return_code": return_code,
            "tail": _tail(log_path),
        },
    }
    metrics_path.write_text(
        json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[fail  ] {variant:13s} {dataset:8s} "
        f"elapsed={elapsed:.1f}s return_code={return_code}"
    )
    return failure


def _nested(report: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _summary_row(
    report: Dict[str, Any], repeat: int
) -> Dict[str, Any]:
    run_summary = report.get("run_summary", {})
    use_run_summary = bool(
        isinstance(run_summary, dict)
        and run_summary.get("completed_epochs", 0)
    )
    result_section = run_summary if use_run_summary else report.get(
        "result", {}
    )
    access_section = (
        run_summary.get("access", {})
        if use_run_summary
        else report.get("access", {})
    )
    remote_edges = (
        _nested(report, "edge_cache", "requests")
        - _nested(report, "edge_cache", "hits")
        if _nested(report, "edge_cache", "requests") > 0
        else access_section.get("remote_edges", 0.0)
    )
    return {
        "variant": report.get("experiment", ""),
        "dataset": report.get("dataset", ""),
        "repeat": repeat,
        "status": _nested(report, "result", "status", default="failed"),
        "loss_is_finite": _nested(
            report, "result", "loss_is_finite", default=False
        ),
        "completed_epochs": run_summary.get("completed_epochs", 0),
        "selected_epoch": _nested(
            report, "evaluation", "epoch", default=""
        ),
        "mean_loss": result_section.get("mean_loss", ""),
        "seconds_per_batch": result_section.get(
            "mean_seconds_per_batch", 0.0
        ),
        "seconds_per_epoch": result_section.get(
            "mean_seconds_per_epoch", 0.0
        ),
        "remote_nodes": access_section.get("remote_nodes", 0.0),
        "remote_edges": remote_edges,
        "remote_edge_bytes": _nested(
            report, "result", "edge_feature_bytes_per_object"
        ) * remote_edges,
        "edge_cache_hit_rate": _nested(
            report, "edge_cache", "object_hit_rate"
        ),
        "edge_cache_bytes_saved": _nested(
            report, "edge_cache", "bytes_saved"
        ),
        "sample_probability_mean": _nested(
            report, "sample_probability", "mean"
        ),
        "negative_mean_hardness": _nested(
            report, "negative_sampler", "mean_hardness"
        ),
        "negative_ess_ratio": _nested(
            report,
            "negative_sampler",
            "effective_sample_size_ratio",
        ),
        "negative_remote_ratio": _nested(
            report, "negative_sampler", "remote_sample_ratio"
        ),
        "negative_reuse_ratio": _nested(
            report, "negative_sampler", "reuse_sample_ratio"
        ),
        "peak_gpu_memory_bytes": _nested(
            report, "result", "peak_gpu_memory_bytes"
        ),
        "validation_ap": _nested(
            report, "evaluation", "validation_ap", default=""
        ),
        "validation_auc": _nested(
            report, "evaluation", "validation_auc", default=""
        ),
        "test_ap": _nested(report, "evaluation", "test_ap", default=""),
        "test_auc": _nested(report, "evaluation", "test_auc", default=""),
        "final_validation_ap": _nested(
            report, "final_evaluation", "validation_ap", default=""
        ),
        "final_validation_auc": _nested(
            report, "final_evaluation", "validation_auc", default=""
        ),
        "final_test_ap": _nested(
            report, "final_evaluation", "test_ap", default=""
        ),
        "final_test_auc": _nested(
            report, "final_evaluation", "test_auc", default=""
        ),
        "log": _nested(report, "runner", "log", default=""),
    }


def _write_summary(
    run_directory: Path,
    reports: Iterable[tuple[int, Dict[str, Any]]],
    *,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows = [_summary_row(report, repeat) for repeat, report in reports]
    csv_path = run_directory / "matrix_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed_cells = {
        (row["variant"], row["dataset"], row["repeat"])
        for row in rows
        if row["status"] == "completed" and row["loss_is_finite"]
    }
    expected_cells = {
        (variant, dataset, repeat)
        for variant in args.variants
        for dataset in DATASETS
        for repeat in range(args.repeats)
    }
    summary = {
        "schema_version": 1,
        "datasets": list(DATASETS),
        "variants": args.variants,
        "batches": args.batches,
        "screen_epochs": args.screen_epochs,
        "dataset_epochs": args.dataset_epochs,
        "repeats": args.repeats,
        "complete": completed_cells == expected_cells,
        "completed_cells": len(completed_cells),
        "expected_cells": len(expected_cells),
        "missing_cells": [
            {"variant": variant, "dataset": dataset, "repeat": repeat}
            for variant, dataset, repeat in sorted(
                expected_cells - completed_cells
            )
        ],
        "rows": rows,
    }
    (run_directory / "matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> int:
    args = _parse_args()
    _validate(args)
    run_directory = args.output_dir / args.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "datasets": list(DATASETS),
        "variants": args.variants,
        "batches": args.batches,
        "screen_epochs": args.screen_epochs,
        "dataset_epochs": args.dataset_epochs,
        "repeats": args.repeats,
        "seed": args.seed,
        "native_library": str(args.native_lib),
        "partitions": {
            dataset: str(_partition_directory(dataset))
            for dataset in DATASETS
        },
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    reports: List[tuple[int, Dict[str, Any]]] = []
    for variant in args.variants:
        for repeat in range(args.repeats):
            for dataset in DATASETS:
                report = _run_one(
                    args,
                    dataset=dataset,
                    variant=variant,
                    repeat=repeat,
                    run_directory=run_directory,
                )
                reports.append((repeat, report))
                _write_summary(run_directory, reports, args=args)

    rows = _write_summary(run_directory, reports, args=args)
    failures = [
        row
        for row in rows
        if row["status"] != "completed" or not row["loss_is_finite"]
    ]
    print(
        f"[matrix] {len(rows) - len(failures)}/{len(rows)} cells completed; "
        f"summary={run_directory / 'matrix_summary.json'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
