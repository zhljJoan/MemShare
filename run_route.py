import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed import DistributedContext
from starrygl.graph import new_vc_route

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


def get_route(bipartite: bool = True):
    ctx = DistributedContext.get_default_context()
    assert ctx.world_size == 3

    dst_ids = torch.tensor(all_nparts[ctx.rank], dtype=torch.long, device=ctx.device)
    edge_index = torch.tensor(all_eparts[ctx.rank], dtype=torch.long, device=ctx.device).t()

    return new_vc_route(dst_ids, edge_index, bipartite=bipartite)

if __name__ == "__main__":
    ctx = DistributedContext.init(backend="gloo", use_gpu=True)
    
    src_ids, edge_index, dst_ids, route = get_route(False)
    src_size = route.src_len
    dst_size = route.dst_len
    ctx.sync_print(route.src_len, route.dst_len)

    edge_ones = torch.ones(edge_index.size(1), device=ctx.device).requires_grad_()
    src_ones = scatter_sum(edge_ones, edge_index[0], dim=0, dim_size=route.src_len)
    dst_ones = scatter_sum(edge_ones, edge_index[1], dim=0, dim_size=route.dst_len)

    # ctx.sync_print(route.fw_tensor(dst_ones))
    # ctx.sync_print(route.bw_tensor(src_ones))

    out = route.reverse_route().apply(src_ones)
    ctx.sync_print(out)

    out.sum().backward()
    ctx.sync_print(edge_ones.grad)

    ctx.sync_print(route.get_src_part_ids())

    ctx.shutdown()
