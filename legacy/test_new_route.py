import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph
from starrygl.core.gather import Gather
from starrygl.core.straight import Straight
from starrygl.parallel import init_process_group, compute_degree
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


def get_distgraph(device = "cpu") -> DistGraph:
    assert dist.get_world_size() == 3
    rank = dist.get_rank()
    ids = torch.tensor(all_nparts[rank], dtype=torch.long, device=device)
    edge_index = torch.tensor(all_eparts[rank], dtype=torch.long, device=device).t()
    return DistGraph(ids, edge_index)

def get_features(device = "cpu") -> Tensor:
    assert dist.get_world_size() == 3
    rank = dist.get_rank()
    return torch.tensor(all_nparts[rank], dtype=torch.float32, device=device) + 1.0

def reduce(g: DistGraph, x: Tensor) -> Tensor:
    edge_index = g.edge_index
    x = x[edge_index[0]]
    x = scatter_sum(x, edge_index[1], dim=0, dim_size=g.dst_size)
    return x

if __name__ == "__main__":
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = init_process_group(backend="nccl")
    
    g = get_distgraph(device)
    # sync_print(g.src_ids.tolist())
    # main_print("="*64)
    
    # sync_print(g.dst_ids.tolist())
    # main_print("="*64)
    
    # sync_print(g.route.forward_routes[0])
    # main_print("="*64)
    
    in_degree, out_degree = compute_degree(g)
    # sync_print(in_degree.tolist())
    # sync_print(out_degree.tolist())
    
    x = get_features(device).requires_grad_()
    # straight = Straight(g.dst_size, num_samples=1)
    # gather = Gather(g.src_size).to(device)

    # sync_print(g)
    # sync_print(gather.get_buffer("last_embd").tolist())
    # main_print("="*64)

    sync_print(x.tolist(), g.route.src_size)
    main_print("="*64)
    
    a = torch.arange(g.route.src_size, dtype=torch.long, device=device)
    # a = None
    
    # z, a = gather(x, None, route=g.route)
    z, a = g.route.apply(x, a, {}, async_op=False)
    sync_print(z.tolist(), "" if a is None else a.tolist(), g.route.dst_size)
    main_print("="*64)
    
    sync_print(g.edge_index[0])
    main_print("="*64)
    
    z = reduce(g, z)
    sync_print(z.tolist())
    main_print("="*64)

    # z = straight(x, a, g)
    # sync_print(z.tolist())
    # main_print("="*64)

    loss = z.sum()
    sync_print(loss.item())
    main_print("="*64)

    loss.backward()
    sync_print(x.grad.tolist())
    main_print("="*64)

    sync_print(f"time_used: {g.route.total_time_used}ms")