import torch
import torch.nn as nn

from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph
from .gcn_conv import GCNConv
from .utils import ShrinkData

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

    def forward(self, g: DistGraph) -> Tuple[Tensor, Tensor]:
        m = g.ndata.get("m")
        if m is None:
            return super().forward(g), None
        
        x = g.ndata["x"]
        gcn_norm = g.edata["gcn_norm"].view(-1, 1)
        
        s = ShrinkData(g, m)
        
        x = x[s.src_mask]
        gcn_norm = gcn_norm[s.edge_mask]
        edge_index = s.edge_index
        
        x = x @ self.weight
        x = x[edge_index[0]] * gcn_norm
        x = scatter_sum(x, edge_index[1], dim=0, dim_size=s.dst_size)
        if self.bias is not None:
            x += self.bias
        return x, s.dst_mask
    
