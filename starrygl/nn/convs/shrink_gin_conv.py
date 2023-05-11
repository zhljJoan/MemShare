import torch
import torch.nn as nn

from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph
from .gin_conv import GINConv
from .utils import ShrinkHelper



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
        
        sh = ShrinkHelper(g, dst_idx)
        
        src_x = x[sh.src_idx]
        dst_x = x[sh.dst_idx]
        edge_index = sh.edges
        
        z = scatter_sum(src_x[edge_index[0]], index=edge_index[1], dim=0, dim_size=sh.dst_size)
        x = z + (1 + self.eps) * dst_x
        return self.nn(x)