#!/usr/bin/env python3
"""Run one isolated baseline/coordinated case on two four-GPU hosts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


POLICIES = {
    "baseline": "boundery_recent_decay",
    "coordinated": "boundery_recent_decay_coordinated",
}
DATASETS = ("WIKI", "LASTFM", "WikiTalk", "Flight", "GDELT", "StackOverflow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=tuple(POLICIES), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=6773)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/tmp/memshare-coordinated-8gpu"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/memshare-coordinated-results"),
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=Path(
            "/home/zlj/SPEED/partition/divided_nodes_seed_starrygl"
        ),
    )
    parser.add_argument("--remote-host", default="10.130.141.33")
    parser.add_argument("--remote-host-key-alias", default="gpu06")
    parser.add_argument("--master-addr", default="192.168.1.107")
    parser.add_argument("--socket-ifname", default="ens4f1np1")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    return args


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
    label = "{}__{}__{}ep__seed{}__g{}".format(
        args.variant,
        args.dataset,
        args.epochs,
        args.seed,
        args.group_size,
    )
    source = args.stage_root / "src"
    examples = source / "examples"
    if not (examples / "train_boundery.py").is_file():
        raise FileNotFoundError(
            "staged source is missing: {}".format(examples)
        )

    logs = args.run_root / "logs"
    metrics_dir = args.run_root / "metrics"
    launcher_dir = args.run_root / "launcher"
    for directory in (logs, metrics_dir, launcher_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "{}.jsonl".format(label)
    local_log = logs / "{}__gpu07.log".format(label)
    remote_log = logs / "{}__gpu06.log".format(label)
    launcher_path = launcher_dir / "{}.json".format(label)
    outputs = (metrics_path, local_log, remote_log, launcher_path)
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(
            "refusing to overwrite: {}".format(
                [str(path) for path in outputs if path.exists()]
            )
        )
    for path in outputs:
        if path.exists():
            path.unlink()

    train_args = [
        "train_boundery.py",
        "--model",
        "TGN_large",
        "--dataname",
        args.dataset,
        "--partition",
        str(args.partition),
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
        "--coordinated_group_size",
        str(args.group_size),
        "--max_epoch",
        str(args.epochs),
        "--freeze_adaptive_parameters",
        "--collect_phase_timing",
        "--metrics_path",
        str(metrics_path),
    ]
    if args.skip_eval:
        train_args.append("--skip_eval")

    environment = {
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
        "NCCL_IB_DISABLE": "1",
        "NCCL_NET": "Socket",
        "NCCL_SOCKET_IFNAME": args.socket_ifname,
        "GLOO_SOCKET_IFNAME": args.socket_ifname,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    torchrun = Path("/home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun")

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
        for key, value in environment.items()
    ]
    remote_command = "cd {} && exec env {} {}".format(
        shlex.quote(str(examples)),
        " ".join(remote_assignments),
        shlex.join(torchrun_command(1)),
    )
    local_environment = os.environ.copy()
    local_environment.update(environment)

    local_before = read_local_nic(args.socket_ifname)
    remote_before = read_remote_nic(args)
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

    local_after = read_local_nic(args.socket_ifname)
    remote_after = read_remote_nic(args)
    local_delta = {
        key: local_after[key] - local_before[key] for key in local_before
    }
    remote_delta = {
        key: remote_after[key] - remote_before[key] for key in remote_before
    }
    endpoint_bytes = sum(local_delta.values()) + sum(remote_delta.values())
    combined_log = local_log.read_text(errors="replace") + remote_log.read_text(
        errors="replace"
    )
    network = {
        "socket_interface_seen": args.socket_ifname in combined_log,
        "socket_network_selected": "Using network Socket" in combined_log,
        "ib_transport_seen": "NET/IB" in combined_log,
    }
    report = {
        "label": label,
        "wall_seconds": time.time() - started,
        "local_return_code": local_process.returncode,
        "remote_return_code": remote_process.returncode,
        "metrics_path": str(metrics_path),
        "local_log": str(local_log),
        "remote_log": str(remote_log),
        "network": network,
        "nic_bytes": {
            "interface": args.socket_ifname,
            "local": local_delta,
            "remote": remote_delta,
            "endpoint_bytes": endpoint_bytes,
            "estimated_wire_bytes": endpoint_bytes / 2.0,
        },
    }
    launcher_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if local_process.returncode or remote_process.returncode:
        return 1
    if not metrics_path.is_file():
        print("metrics file was not produced", file=sys.stderr)
        return 1
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    if len(records) != args.epochs or any(
        record.get("world_size") != 8 for record in records
    ):
        print("epoch metrics are incomplete", file=sys.stderr)
        return 1
    if not all(
        (
            network["socket_interface_seen"],
            network["socket_network_selected"],
            not network["ib_transport_seen"],
        )
    ):
        print("NCCL Socket route validation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
