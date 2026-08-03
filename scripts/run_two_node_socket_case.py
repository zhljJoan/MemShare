#!/usr/bin/env python3
"""Run one paired-study case on gpu07 + gpu06 over the 10 GbE socket path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


DATASETS = (
    "WIKI",
    "LASTFM",
    "WikiTalk",
    "Flight",
    "GDELT",
    "StackOverflow",
)
POLICIES = {
    "baseline": "boundery_recent_decay",
    "coordinated-decay": "boundery_recent_decay_coordinated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=tuple(POLICIES), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batches", type=int)
    mode.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=6773)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/tmp/memshare_core_8gpu_20260803"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/home/zlj/MemShare-public/MemShare/outputs/"
            "core_memshare_8gpu_socket/20260803"
        ),
    )
    parser.add_argument("--remote-host", default="10.130.141.33")
    parser.add_argument("--remote-host-key-alias", default="gpu06")
    parser.add_argument("--master-addr", default="192.168.1.107")
    parser.add_argument("--socket-ifname", default="ens4f1np1")
    parser.add_argument("--label-suffix", default="")
    parser.add_argument("--no-phase-timing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--feature-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()
    count = parsed.batches if parsed.batches is not None else parsed.epochs
    if count is None or count <= 0:
        parser.error("batch/epoch count must be positive")
    if not 1024 <= parsed.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    return parsed


def terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def read_local_nic(interface: str) -> dict[str, int]:
    statistics = Path("/sys/class/net") / interface / "statistics"
    return {
        "rx_bytes": int((statistics / "rx_bytes").read_text().strip()),
        "tx_bytes": int((statistics / "tx_bytes").read_text().strip()),
    }


def ssh_command(args: argparse.Namespace, command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "HostKeyAlias={}".format(args.remote_host_key_alias),
        args.remote_host,
        command,
    ]


def read_remote_nic(args: argparse.Namespace) -> dict[str, int]:
    statistics = "/sys/class/net/{}/statistics".format(args.socket_ifname)
    output = subprocess.check_output(
        ssh_command(
            args,
            "cat {}/rx_bytes {}/tx_bytes".format(statistics, statistics),
        ),
        text=True,
    ).split()
    return {"rx_bytes": int(output[0]), "tx_bytes": int(output[1])}


def main() -> int:
    args = parse_args()
    mode_label = (
        "{}b".format(args.batches)
        if args.batches is not None
        else "{}ep".format(args.epochs)
    )
    suffix = "__{}".format(args.label_suffix) if args.label_suffix else ""
    label = "{}__{}__{}__seed{}{}".format(
        args.variant, args.dataset, mode_label, args.seed, suffix
    )
    logs = args.run_root / "logs"
    metrics_dir = args.run_root / "metrics"
    launcher_dir = args.run_root / "launcher"
    for directory in (logs, metrics_dir, launcher_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "{}.json".format(label)
    local_log = logs / "{}__gpu07.log".format(label)
    remote_log = logs / "{}__gpu06.log".format(label)
    launcher_path = launcher_dir / "{}.json".format(label)
    outputs = (metrics_path, local_log, remote_log, launcher_path)
    if not args.overwrite and any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(
            "refusing to overwrite existing outputs: {}".format(existing)
        )

    source = args.stage_root / "src"
    examples = source / "examples"
    sampler_library = args.stage_root / "build/libstarrygl_sampler.so"
    partition = (
        args.stage_root / "partitions"
        if args.dataset == "Flight"
        else Path("/home/zlj/SPEED/partition/divided_nodes_seed_starrygl")
    )
    torchrun = Path("/home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun")
    train_args = [
        "train_boundery.py",
        "--model",
        "TGN_large",
        "--dataname",
        args.dataset,
        "--partition",
        str(partition),
        "--topk",
        "0.1",
        "--sample_type",
        POLICIES[args.variant],
        "--probability",
        "0.1",
        "--memory_type",
        "historical",
        "--shared_memory_ssim",
        "0.3",
        "--seed",
        str(args.seed),
        "--core_group_size",
        str(args.group_size),
        "--core_freeze_adaptive_parameters",
        "--core_metrics_path",
        str(metrics_path),
        "--core_experiment_name",
        label,
        "--core_feature_device",
        args.feature_device,
        "--core_nic_ifname",
        args.socket_ifname,
    ]
    if not args.no_phase_timing:
        train_args.append("--core_collect_phase_timing")
    if args.skip_eval:
        train_args.append("--skip_eval")
    if args.batches is not None:
        train_args.extend(["--core_smoke_batches", str(args.batches)])
    else:
        train_args.extend(["--core_screen_epochs", str(args.epochs)])

    common_environment = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "CUDA_HOME": "/home/zlj/local/cuda-11.8",
        "METIS_DLL": (
            "/home/zlj/local/metis-5.1.0/build/Linux-x86_64/"
            "libmetis/libmetis.so"
        ),
        "LD_LIBRARY_PATH": (
            "/home/zlj/.miniconda3/envs/tgnn_3.10/lib:"
            "/home/zlj/local/cuda-11.8/lib64"
        ),
        "PYTHONPATH": str(source),
        "STARRYGL_SAMPLER_LIBRARY": str(sampler_library),
        "NCCL_IB_DISABLE": "1",
        "NCCL_NET": "Socket",
        "NCCL_SOCKET_IFNAME": args.socket_ifname,
        "GLOO_SOCKET_IFNAME": args.socket_ifname,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }

    def torchrun_command(node_rank: int) -> list[str]:
        return [
            str(torchrun),
            "--nnodes=2",
            "--nproc_per_node=4",
            "--node_rank={}".format(node_rank),
            "--master_addr={}".format(args.master_addr),
            "--master_port={}".format(args.port),
            *train_args,
        ]

    remote_assignments = [
        "{}={}".format(key, shlex.quote(value))
        for key, value in common_environment.items()
    ]
    remote_command = "cd {} && exec env {} {}".format(
        shlex.quote(str(examples)),
        " ".join(remote_assignments),
        shlex.join(torchrun_command(1)),
    )
    local_environment = os.environ.copy()
    local_environment.update(common_environment)

    local_nic_before = read_local_nic(args.socket_ifname)
    remote_nic_before = read_remote_nic(args)
    started = time.time()
    with remote_log.open("wb") as remote_output, local_log.open("wb") as local_output:
        remote_process = subprocess.Popen(
            ssh_command(args, remote_command),
            stdout=remote_output,
            stderr=subprocess.STDOUT,
        )
        local_process = subprocess.Popen(
            torchrun_command(0),
            cwd=examples,
            env=local_environment,
            stdout=local_output,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                local_return = local_process.poll()
                remote_return = remote_process.poll()
                if local_return is not None and remote_return is not None:
                    break
                if local_return not in (None, 0) and remote_return is None:
                    terminate(remote_process)
                if remote_return not in (None, 0) and local_return is None:
                    terminate(local_process)
                time.sleep(0.2)
        except BaseException:
            terminate(local_process)
            terminate(remote_process)
            raise
    elapsed = time.time() - started
    local_nic_after = read_local_nic(args.socket_ifname)
    remote_nic_after = read_remote_nic(args)
    local_nic_delta = {
        key: local_nic_after[key] - local_nic_before[key]
        for key in local_nic_before
    }
    remote_nic_delta = {
        key: remote_nic_after[key] - remote_nic_before[key]
        for key in remote_nic_before
    }
    endpoint_bytes = sum(local_nic_delta.values()) + sum(
        remote_nic_delta.values()
    )

    local_text = local_log.read_text(encoding="utf-8", errors="replace")
    remote_text = remote_log.read_text(encoding="utf-8", errors="replace")
    combined_log = local_text + "\n" + remote_text
    network_check = {
        "bootstrap_local_10gbe": (
            "Bootstrap : Using {}:192.168.1.107".format(args.socket_ifname)
            in local_text
        ),
        "bootstrap_remote_10gbe": (
            "Bootstrap : Using {}:192.168.1.106".format(args.socket_ifname)
            in remote_text
        ),
        "socket_network_selected": "Using network Socket" in combined_log,
        "ib_transport_seen": "NET/IB" in combined_log,
    }
    report = {
        "label": label,
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "group_size": args.group_size,
        "wall_seconds": elapsed,
        "local_return_code": local_process.returncode,
        "remote_return_code": remote_process.returncode,
        "nic_bytes": {
            "interface": args.socket_ifname,
            "local": local_nic_delta,
            "remote": remote_nic_delta,
            "endpoint_bytes": endpoint_bytes,
            "estimated_wire_bytes": endpoint_bytes / 2.0,
            "scope": (
                "whole-interface counters; includes protocol overhead and any "
                "unrelated traffic during the run"
            ),
        },
        "metrics_path": str(metrics_path),
        "local_log": str(local_log),
        "remote_log": str(remote_log),
        "network": network_check,
    }
    launcher_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if local_process.returncode or remote_process.returncode:
        return 1
    if not metrics_path.exists():
        print("metrics file was not produced", file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("world_size") != 8:
        print("metrics did not aggregate eight ranks", file=sys.stderr)
        return 1
    training_network = metrics.get("network_bytes", {})
    if not (
        training_network.get("enabled")
        and training_network.get("interface") == args.socket_ifname
        and training_network.get("host_count") == 2
    ):
        print(
            "training-window NIC counters did not cover both hosts",
            file=sys.stderr,
        )
        return 1
    if not all(
        (
            network_check["bootstrap_local_10gbe"],
            network_check["bootstrap_remote_10gbe"],
            network_check["socket_network_selected"],
            not network_check["ib_transport_seen"],
        )
    ):
        print("NCCL network-route validation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
