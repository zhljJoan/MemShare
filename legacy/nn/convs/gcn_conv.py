import torch
import torch.nn as nn

from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph

class GCNConv(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g: DistGraph) -> Tensor:
        x = g.ndata["x"]
        gcn_norm = g.edata["gcn_norm"].view(-1, 1)
        edge_index = g.edge_index

        x = x @ self.weight
        x = x[edge_index[0]] * gcn_norm
        x = scatter_sum(x, edge_index[1], dim=0, dim_size=g.dst_size)
        if self.bias is not None:
            x += self.bias
        return x
