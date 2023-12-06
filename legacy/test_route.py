import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.core.route import Route
from starrygl.graph.utils import init_local_edge_index
from starrygl.parallel import init_process_group
from starrygl.utils.printer import main_print, sync_print

from torch_scatter import scatter_sum

all_nparts = [
    [0, 1],
    [2, 3, 4],
    [5, 6],
]

all_eparts = [
    [
        [2, 0], [1, 0], [0, 1], [3, 1], [4, 1],
    ],
    [
        [0, 2], [3, 2], [5, 2], [1, 3], [2, 3],
        [4, 3], [6, 3], [1, 4], [3, 4],
    ],
    [
        [2, 5], [6, 5], [5, 6], [3, 6],
    ],
]


def get_route(device = "cpu") -> Tuple[Route, Tensor]:
    assert dist.get_world_size() == 3
    rank = dist.get_rank()
    ids = torch.tensor(all_nparts[rank], dtype=torch.long, device=device)
    edge_index = torch.tensor(all_eparts[rank], dtype=torch.long, device=device).t()

    src_ids, local_edge_index = init_local_edge_index(ids, edge_index)
    return Route(ids, src_ids), local_edge_index


def get_features(device = "cpu") -> Tensor:
    assert dist.get_world_size() == 3
    rank = dist.get_rank()
    return torch.tensor(all_nparts[rank], dtype=torch.float32, device=device) + 1.0


def reduce(x: Tensor, edge_index: Tensor, dim_size: int) -> Tensor:
    x = x[edge_index[0]]
    x = scatter_sum(x, edge_index[1], dim=0, dim_size=dim_size)
    return x

if __name__ == "__main__":
    device = init_process_group(backend="mpi")
    
    route, edge_index = get_route(device)
    src_size = route.dst_size
    dst_size = route.src_size

    # in_degree, out_degree = compute_degree(g)
    # sync_print(in_degree.tolist())
    # sync_print(out_degree.tolist())
    
    x = get_features(device).requires_grad_()
    sync_print(x.tolist())
    main_print("="*64)

    z, _ = route.apply(x, None)
    sync_print(z.tolist())
    main_print("="*64)

    h = reduce(z, edge_index, dst_size)
    sync_print(h.tolist())
    main_print("="*64)

    h.sum().backward()
    sync_print(x.grad.tolist())
    main_print("="*64)
