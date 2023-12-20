import torch

from torch import Tensor
from typing import *

__all__ = [
    "init_vc_edge_index",
]

def init_vc_edge_index(
    dst_ids: Tensor,
    edge_index: Tensor,
    bipartite: bool = True,
) -> Tuple[Tensor, Tensor]:
    ikw = dict(dtype=torch.long, device=dst_ids.device)

    local_num_nodes = torch.zeros(1, **ikw)
    if dst_ids.numel() > 0:
        local_num_nodes = dst_ids.max().max(local_num_nodes)
    if edge_index.numel() > 0:
        local_num_nodes = edge_index.max().max(local_num_nodes)
    local_num_nodes = local_num_nodes.item() + 1

    xmp: Tensor = torch.zeros(local_num_nodes, **ikw)
    xmp[edge_index[1].unique()] += 0b01
    xmp[dst_ids.unique()] += 0b10
    if not (xmp != 0x01).all():
        raise RuntimeError(f"must be vertex-cut partition graph")
    
    if bipartite:
        src_ids = edge_index[0].unique()
    else:
        xmp.fill_(0)
        xmp[edge_index[0]] = 1
        xmp[dst_ids] = 0
        src_ids = torch.cat([dst_ids, torch.where(xmp > 0)[0]], dim=-1)

    xmp.fill_((2**62-1)*2+1)
    xmp[src_ids] = torch.arange(src_ids.size(0), **ikw)
    src = xmp[edge_index[0]]
    
    xmp.fill_((2**62-1)*2+1)
    xmp[dst_ids] = torch.arange(dst_ids.size(0), **ikw)
    dst = xmp[edge_index[1]]
    
    local_edge_index = torch.vstack([src, dst])
    return src_ids, local_edge_index
