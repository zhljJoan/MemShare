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

class GINConv(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        mlp_channels: Optional[int] = None,
        eps: float = 0,
        train_eps: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mlp_channels is None:
            mlp_channels = in_channels + out_channels
        self.mlp_channels = mlp_channels
        self.nn = nn.Sequential(
            nn.Linear(in_channels, mlp_channels),
            nn.ReLU(),
            nn.Linear(mlp_channels, out_channels),
        )
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.tensor[eps])
        else:
            self.register_buffer("eps", torch.tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        pyg_inits.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, g: DistGraph, x: Tensor) -> Tuple[Tensor, Tensor]:
        edge_index = g.edata[EID]
        t = x[edge_index[0]]
        t = scatter_sum(t, edge_index[1], dim=0, dim_size=g.ndata.dst_size)
        x = t + (1 + self.eps) * x[:g.ndata.dst_size]
        return self.nn(x)
