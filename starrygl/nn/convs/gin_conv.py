import torch
import torch.nn as nn

from torch_scatter import scatter_sum

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph

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
        self.initial_eps = eps

        self.nn = nn.Sequential(
            nn.Linear(in_channels, mlp_channels),
            nn.ReLU(),
            nn.Linear(mlp_channels, out_channels),
        )

        if train_eps:
            self.eps = torch.nn.Parameter(torch.tensor[eps])
        else:
            self.register_buffer("eps", torch.tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.nn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            if "bias" in name:
                nn.init.zeros_(param)
        nn.init.constant_(self.eps, self.initial_eps)

    def forward(self, g: DistGraph) -> Tuple[Tensor, Tensor]:
        x = g.ndata["x"]
        edge_index = g.edge_index
        z = x[edge_index[0]]
        z = scatter_sum(z, edge_index[1], dim=0, dim_size=g.dst_size)
        x = z + (1 + self.eps) * x[:g.dst_size]
        return self.nn(x)