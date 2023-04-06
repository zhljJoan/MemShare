import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.inits as pyg_inits
import torch_geometric.utils as pyg_utils
from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph, EID

class GATConv(nn.Module):
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
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        self.lin = pyg_nn.Linear(in_channels, heads * out_channels,
                                 bias=False, weight_initializer="glorot")
        
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = pyg_nn.Linear(edge_dim, heads * out_channels,
                                          bias=False, weight_initializer="glorot")
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        pyg_inits.glorot(self.att_src)
        pyg_inits.glorot(self.att_dst)

        if hasattr(self, "lin_edge"):
            self.lin_edge.reset_parameters()
            pyg_inits.glorot(self.att_edge)
        
        if hasattr(self, "bias"):
            pyg_inits.zeros(self.bias)

    def forward(self, g: DistGraph, x: Tensor) -> Tuple[Tensor, Tensor]:
        H, C = self.heads, self.out_channels
        edge_index = g.edata[EID]

        x = self.lin(x).view(-1, H, C)
        
        alpha_j = (x * self.att_src).sum(dim=-1)
        alpha_j = alpha_j[edge_index[0]]

        alpha_i = (x * self.att_dst).sum(dim=-1)
        alpha_i = alpha_i[edge_index[1]]

        if hasattr(self, "lin_edge"):
            edge_attr = g.edata["edge_attr"]
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            e = self.lin_edge(edge_attr).view(-1, H, C)
            alpha_e = (e * self.att_edge).sum(dim=-1)
            alpha = alpha_i + alpha_j + alpha_e
        else:
            alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = pyg_utils.softmax(
            src=alpha,
            index=edge_index[1],
            num_nodes=g.ndata.dst_size,
        )
        if self.dropout != 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        x = alpha.unsqueeze(-1) * x[edge_index[0]]
        x = scatter_sum(x, edge_index[1], dim=0, dim_size=g.ndata.dst_size)

        if self.concat:
            x = x.view(-1, self.heads * self.out_channels)
        else:
            x = x.mean(dim=1)
        
        if hasattr(self, "bias"):
            x += self.bias
        return x

