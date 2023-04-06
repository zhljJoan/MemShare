import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

import copy
import inspect

from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from ..graph.distgraph import DistGraph
from ..core.gather import Gather

class BasicGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Optional[str] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        out_channels = hidden_channels \
            if out_channels is None else out_channels
        self.out_channels = out_channels

        self.dropout = dropout
        
        self.gathers: List[Gather] = []
        for _ in range(num_layers):
            self.gathers.append(Gather())

        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            in_channels = hidden_channels
        
        if jk is None:
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            
            if jk != "last":
                self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)
        
            if jk == "cat":
                jk_channels = num_layers * hidden_channels
            else:
                jk_channels = hidden_channels
            self.lin_jk = nn.Linear(jk_channels, out_channels)

        self.act = activation_resolver(act, **(act_kwargs or {}))
        if norm is not None:
            self.norms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(normalization_resolver(
                    norm, hidden_channels, **(norm_kwargs or {})))
            if jk is not None:
                self.norms.append(normalization_resolver(
                    norm, hidden_channels, **(norm_kwargs or {})))
        
        self.reset_parameters()

    def init_conv(self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ) -> nn.Module:
        raise NotImplementedError

    def reset_parameters(self):
        if hasattr(self, "jk"):
            self.jk.reset_parameters()
            
        if hasattr(self, "lin_jk"):
            self.lin_jk.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
        
        if hasattr(self, "norms"):
            for norm in self.norms:
                norm.reset_parameters()

    def forward(
        self,
        g: DistGraph,
    ) -> Tensor:
        xs: List[Tensor] = []

        x = g.ndata["x"]
        async_op = g.args.get("async_op", False)
        for i in range(self.num_layers):
            # 获取邻居信息
            x, _ = self.gathers[i](x, None, g.route, async_op=async_op)
            
            # ！在考虑这个dropout层的位置
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 执行图卷积
            x = self.convs[i](g, x)

            # 如果没有jk，则最后一层不加bn和act
            if i == self.num_layers - 1 and not hasattr(self, "jk"):
                break

            if hasattr(self, "norms"):
                x = self.norms[i](x)
            if hasattr(self, "act"):
                x = self.act(x) 
            if hasattr(self, "jk"):
                xs.append(x)
        
        if hasattr(self, "jk"):
            x = self.jk([t for t in xs])
        x = self.lin_jk(x) if hasattr(self, "lin_jk") else x
        return x
