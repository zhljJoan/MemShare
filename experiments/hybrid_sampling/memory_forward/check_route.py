"""Exercise state replacement and invalidation on GPU without a distributed launch."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json

import torch
from starrygl.distributed.context import DistributedContext
from starrygl.sample import data_loader
from starrygl.sample.memory.shared_mailbox import SharedMailBox
from route import install


def field(tensor):
    return SimpleNamespace(accessor=SimpleNamespace(data=tensor))


def check(directory):
    install("forward", directory)
    gpu = torch.device("cuda:0")
    box = SharedMailBox.__new__(SharedMailBox)
    box.num_parts = 1
    box.node_memory = field(torch.zeros(7, 3, device=gpu))
    box.node_memory_ts = field(torch.zeros(7, device=gpu))
    box.mailbox = field(torch.randn(7, 2, 5, device=gpu))
    box.mailbox_ts = field(torch.randn(7, 2, device=gpu))
    box.is_shared_mask = torch.tensor([-1, -1, -1, -1, -1, 0, -1], device=gpu)
    box.shared_nodes_index = torch.tensor([5], device=gpu)
    box.historical_cache = SimpleNamespace(synchronize_shared_update=lambda: (
        torch.tensor([0], device=gpu), torch.full((1, 3), 50., device=gpu),
        torch.tensor([100.], device=gpu)))
    graph = SimpleNamespace(feature_device=gpu, nfeat=field(torch.randn(7, 4, device=gpu)))
    graph.nfeat.all_to_all_ind2ptr = lambda ids, group: {"recv_ind": ids}
    query = torch.tensor([5, 1, 3, 1, 6], device=gpu)
    index = torch.tensor([3, 5, 1, 3], device=gpu)
    source = torch.arange(12., device=gpu).reshape(4, 3)
    times = torch.ones(4, device=gpu)

    def read(ids, expected_hits):
        data_loader.get_node_all_to_all_route(graph, box, ids, gpu)
        last = Path(directory, "rank-0.jsonl").read_text().splitlines()[-1]
        assert json.loads(last)["hits"] == expected_hits

    box.set_memory_local(index, source, times)
    box.sychronize_shared()
    read(query, 2)  # Shared row 5 and duplicate writer 3 use authoritative storage.
    read(query, 0)  # Only the immediately following route may use this buffer.
    box.set_memory_local(index, source, times)
    source.add_(7)
    read(query, 0)
    box.set_memory_local(index, source, times)
    box.node_memory.accessor.data[1].fill_(-1)
    read(query, 0)
    read(query[:0], 0)


if __name__ == "__main__":
    with TemporaryDirectory() as directory, patch.object(
        DistributedContext, "get_default_context", return_value=SimpleNamespace(memory_nccl_group=None)
    ), patch("torch.distributed.get_rank", return_value=0):
        check(directory)
    print("forwarding, invalidation and empty route checks passed")
