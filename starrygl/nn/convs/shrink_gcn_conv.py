import torch
import torch.nn as nn

from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph
from .gcn_conv import GCNConv
from .utils import ShrinkHelper

class ShrinkGCNConv(GCNConv):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            **kwargs
        )

    def forward(self,
        g: DistGraph,
        sh: Optional[ShrinkHelper] = None,
        dst_idx: Optional[Tensor] = None,
    ) -> Tensor:
        if sh is None and dst_idx is None:
            return super().forward(g)
        
        if sh is None:
            sh = ShrinkHelper(g, dst_idx)
        
        x = g.ndata["x"]
        gcn_norm = g.edata["gcn_norm"].view(-1, 1)
                
        x = x[sh.src_idx]
        gcn_norm = gcn_norm[sh.edge_idx]
        edge_index = sh.edges
        
        x = x @ self.weight
        x = x[edge_index[0]] * gcn_norm
        x = scatter_sum(x, edge_index[1], dim=0, dim_size=sh.dst_size)
        if self.bias is not None:
            x += self.bias
        return x
    
