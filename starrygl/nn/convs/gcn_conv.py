import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph, EID

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
        self.lin = pyg_nn.Linear(in_channels, out_channels,
                                 bias=False, weight_initializer="glorot")
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, g: DistGraph, x: Tensor) -> Tuple[Tensor, Tensor]:
        edge_index = g.edata[EID]
        x = self.lin(x)
        x = x[edge_index[0]]
        x = x * g.edata["gcn_norm"].view(-1, 1)
        x = scatter_sum(x, edge_index[1], dim=0, dim_size=g.ndata.dst_size)
        if self.bias is not None:
            x += self.bias
        return x
