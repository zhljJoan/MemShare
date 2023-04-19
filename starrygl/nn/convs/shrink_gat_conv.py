import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph
from .gat_conv import GATConv
from .utils import ShrinkData



class ShrinkGATConv(GATConv):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=bias,
            **kwargs
        )

    def forward(self, g: DistGraph) -> Tuple[Tensor, Tensor]:
        m = g.ndata.get("m")
        if m is None:
            return super().forward(g), None
        
        H, C = self.heads, self.out_channels
        x = g.ndata["x"]
        
        s = ShrinkData(g, m)
        x = x[s.src_mask]
        edge_index = s.edge_index

        x = (x @ self.weight).view(-1, H, C)
        
        alpha_j = (x * self.att_src).sum(dim=-1)
        alpha_j = alpha_j[edge_index[0]]

        alpha_i = (x * self.att_dst).sum(dim=-1)
        alpha_i = alpha_i[edge_index[1]]

        if self.edge_dim is not None:
            edge_attr = g.edata["edge_attr"]
            edge_attr = edge_attr[s.edge_mask]
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            
            e = (edge_attr @ self.lin_edge).view(-1, H, C)
            alpha_e = (e * self.att_edge).sum(dim=-1)
            alpha = alpha_i + alpha_j + alpha_e
        else:
            alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(
            src=alpha,
            index=edge_index[1],
            num_nodes=s.dst_size,
        )

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        x = x[edge_index[0]] * alpha.view(-1, H, 1)
        x = scatter_sum(x, edge_index[1], dim=0, dim_size=s.dst_size)

        if self.concat:
            x = x.view(-1, H * C)
        else:
            x = x.mean(dim=1)
        
        if self.bias is not None:
            x += self.bias
        return x, s.dst_mask

