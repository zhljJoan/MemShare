import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph

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
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = nn.Parameter(torch.Tensor(edge_dim, heads * out_channels))
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.att_src)
        nn.init.xavier_normal_(self.att_dst)

        if self.edge_dim is not None:
            nn.init.xavier_normal_(self.lin_edge)
            nn.init.xavier_normal_(self.att_edge)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g: DistGraph) -> Tuple[Tensor, Tensor]:
        H, C = self.heads, self.out_channels
        x = g.ndata["x"]
        edge_index = g.edge_index

        x = (x @ self.weight).view(-1, H, C)
        
        alpha_j = (x * self.att_src).sum(dim=-1)
        alpha_j = alpha_j[edge_index[0]]

        alpha_i = (x * self.att_dst).sum(dim=-1)
        alpha_i = alpha_i[edge_index[1]]

        if self.edge_dim is not None:
            edge_attr = g.edata["edge_attr"]
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
            num_nodes=g.dst_size,
        )

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        x = x[edge_index[0]] * alpha.view(-1, H, 1)
        x = scatter_sum(x, edge_index[1], dim=0, dim_size=g.dst_size)

        if self.concat:
            x = x.view(-1, H * C)
        else:
            x = x.mean(dim=1)
        
        if self.bias is not None:
            x += self.bias
        return x

