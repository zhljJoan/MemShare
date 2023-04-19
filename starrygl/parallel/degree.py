import torch

from torch import Tensor
from typing import *

from torch_scatter import scatter_sum

from ..graph.distgraph import DistGraph
from ..core.gather import Gather


def compute_degree(g: DistGraph) -> Tuple[Tensor, Tensor]:
    edge_index = g.edge_index
    x = torch.ones(
        edge_index.size(1),
        dtype=torch.long,
        device=edge_index.device,
    )
    in_degree = scatter_sum(x, edge_index[1], dim=0, dim_size=g.dst_size) # (g.dst_size,)
    out_degree = scatter_sum(x, edge_index[0], dim=0, dim_size=g.src_size)
    out_degree = g.gather(out_degree, direct="src_to_dst") # (g.dst_size,)
    
    in_degree = g.gather(in_degree, direct="dst_to_src") # (g.src_size,)
    out_degree = g.gather(out_degree, direct="dst_to_src") # (g.src_size,)
    return in_degree, out_degree

def compute_gcn_norm(g: DistGraph) -> Tensor:
    edge_index = g.edge_index
    in_deg, out_deg = compute_degree(g)
    
    a = in_deg[edge_index[0]].pow(-0.5)
    b = out_deg[edge_index[0]].pow(-0.5)
    x = a * b
    x[x.isinf()] = 0.0
    x[x.isnan()] = 0.0
    return x
