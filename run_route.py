import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed import DistributedContext
from starrygl.data import *

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

def get_data():
    ctx = DistributedContext.get_default_context()
    assert ctx.world_size == 3

    dst_ids = torch.tensor(all_nparts[ctx.rank], dtype=torch.long, device=ctx.device)
    edge_index = torch.tensor(all_eparts[ctx.rank], dtype=torch.long, device=ctx.device).t()

    src_ids, edge_index = init_vc_edge_index(dst_ids, edge_index)
    return GraphData.from_bipartite(edge_index, raw_src_ids=src_ids, raw_dst_ids=dst_ids)

if __name__ == "__main__":
    ctx = DistributedContext.init(backend="gloo", use_gpu=True)
    
    g = get_data()
    route = g.to_route()
    edge_index = g.edge_index()

    # src_ids, edge_index, dst_ids, route = get_route(False)
    # src_size = route.src_len
    # dst_size = route.dst_len
    ctx.sync_print(route.src_len, route.dst_len)
    ctx.sync_print(route._fw_ptr, route._fw_ind)
    ctx.sync_print(route._bw_ptr, route._bw_ind)

    edge_ones = torch.ones(edge_index.size(1), device=ctx.device).requires_grad_()
    src_ones = scatter_sum(edge_ones, edge_index[0], dim=0, dim_size=route.src_len)
    dst_ones = scatter_sum(edge_ones, edge_index[1], dim=0, dim_size=route.dst_len)

    ctx.sync_print(route.fw_tensor(dst_ones))
    ctx.sync_print(route.bw_tensor(src_ones))

    out = route.rev().apply(src_ones)
    ctx.sync_print(out)

    out.sum().backward()
    ctx.sync_print(edge_ones.grad)

    ctx.sync_print(route.get_src_part_ids())

    dst_mask = torch.full((route.dst_len,), ctx.rank % 2, dtype=torch.bool, device=ctx.device)
    ctx.main_print("="*64)
    ctx.sync_print(dst_mask)
    _, _, r2 = route.filter(dst_mask)
    ctx.sync_print(r2.apply(dst_ones).detach())
    ctx.sync_print(r2.rev().apply(src_ones).detach())
    # dst_true = torch.ones(route.dst_len, dtype=torch.float, device=ctx.device)
    # ctx.sync_print(route.fw_tensor(dst_true, "max"))

    spb = g.to_sparse()
    x = torch.randn(g.node("dst").num_nodes, 64, device=ctx.device).requires_grad_()
    ctx.sync_print(x[:,0])

    y = spb.apply(x)
    ctx.sync_print(y[:,0])

    y.sum().backward()
    ctx.sync_print(x.grad[:,0])

    ctx.shutdown()
