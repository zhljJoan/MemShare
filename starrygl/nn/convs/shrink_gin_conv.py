import torch
import torch.nn as nn

from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph
from .gin_conv import GINConv
from .utils import ShrinkData



class ShrinkGINConv(GINConv):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        mlp_channels: Optional[int] = None,
        eps: float = 0,
        train_eps: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_channels=mlp_channels,
            eps=eps,
            train_eps=train_eps,
            **kwargs
        )

    def forward(self, g: DistGraph) -> Tuple[Tensor, Tensor]:
        m = g.ndata.get("m")
        if m is None:
            return super().forward(g), None
        
        x = g.ndata["x"]
        
        s = ShrinkData(g, m)
        edge_index = s.edge_index
        
        z = x[s.src_mask][edge_index[0]]
        z = scatter_sum(z, edge_index[1], dim=0, dim_size=s.dst_size)
        x = z + (1 + self.eps) * x[s.dst_mask]
        return self.nn(x), s.dst_mask