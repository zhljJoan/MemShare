"""Compare edge IDs and root occurrence order to the installed MemShare sampler."""

import numpy as np
import torch

from probe import FixedWorkspace, SplitSampler
from starrygl import sampler_ops as ops


def main():
    rng = np.random.default_rng(27)
    ptr = np.array([0, 0, 1, 80, 205, 211, 800])
    neighbor = rng.integers(6, size=ptr[-1])
    owner = np.repeat(np.arange(6), np.diff(ptr))
    times = np.concatenate([np.sort(rng.integers(1, 51, size=d)) for d in np.diff(ptr)])
    edge_part = rng.integers(2, size=len(times), dtype=np.int32)
    node_part = np.arange(6, dtype=np.int32) % 2
    remote = (edge_part != 0) | (node_part[neighbor] != 0)
    adj = ops.get_neighbors(
        "hybrid-self-check",
        torch.from_numpy(neighbor),
        torch.from_numpy(owner),
        6,
        0,
        torch.arange(len(times)),
        None,
        None,
        torch.from_numpy(times),
    )
    for policy in ["recent", "boundery_recent_decay_coordinated"]:
        coordinated = policy != "recent"
        native = ops.ParallelSampler(
            adj,
            6,
            len(times),
            2,
            [20],
            1,
            policy,
            0,
            torch.from_numpy(edge_part),
            torch.from_numpy(node_part),
            0.1,
        )
        if coordinated:
            native.configure_coordinated_sampling(8, 6773)
        samplers = [
            SplitSampler(
                ptr,
                neighbor,
                times,
                remote,
                [0, 1, 3, 5],
                coordinated=coordinated,
                workspace=workspace,
            )
            for workspace in (
                None,
                FixedWorkspace(20, coordinated),
                FixedWorkspace(20, coordinated, replay=True),
            )
        ]
        for batch in range(24):
            roots = rng.integers(6, size=[1, 128, 511, 1537, 2, 64][batch % 6])
            stamp = rng.integers(0, 54, size=len(roots))
            beta = [0.0, 0.1, 0.5, 1.0][batch % 4]
            native.neighbor_sample_from_nodes(
                torch.from_numpy(roots), torch.from_numpy(stamp), None, beta
            )
            block = native.get_ret()[0]
            for split in samplers:
                result, _ = split.sample(roots, stamp, batch=batch, beta=beta)
                row, slot = np.where(result >= 0)
                np.testing.assert_array_equal(row, block.src_index().numpy())
                np.testing.assert_array_equal(result[row, slot], block.eid().numpy())
    print("48 batches match native root occurrence order and edge IDs")


if __name__ == "__main__":
    main()
