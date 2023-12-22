# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch import Tensor
# from typing import *

# import copy
# import inspect

# from torch_geometric.nn import JumpingKnowledge
# from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

# from dataclasses import dataclass

# from ..core.cache import NodeProbe
# from ..graph.distgraph import DistGraph
# # from ..core.gather import Gather
# # from ..core.straight import Straight

# @dataclass
# class BasicInputOptions:
#     weight: bool = True
#     bias: bool = True
#     gather: bool = True
#     gather_first: bool = False
#     dropout: float = 0.0
#     act: Optional[str] = None
#     act_kwargs: Optional[Dict[str, Any]] = None
#     act_first: bool = False
#     norm: Optional[str] = None
#     norm_kwargs: Optional[Dict[str, Any]] = None
#     straight_enabled: bool = True
#     straight_num_samples: Optional[int] = None

# @dataclass
# class BasicLayerOptions:
#     in_channels: int
#     hidden_channels: int
#     num_layers: int
#     out_channels: Optional[int] = None
#     gather_beta: float = 1.0
#     dropout: float = 0.0
#     act: Optional[str] = "relu"
#     act_kwargs: Optional[Dict[str, Any]] = None
#     act_first: bool = False
#     norm: Optional[str] = None
#     norm_kwargs: Optional[Dict[str, Any]] = None
#     jk_mode: Optional[str] = None

# @dataclass
# class BasicStraightOptions:
#     enabled: bool = False
#     num_samples: Optional[int] = None
#     beta: float = 1.0

# class BasicGNN(nn.Module):
#     def __init__(
#         self,
#         g: DistGraph,
#         layer_options: BasicLayerOptions,
#         input_options: BasicInputOptions = BasicInputOptions(),
#         straight_options: BasicStraightOptions = BasicStraightOptions(),
#         **kwargs,
#     ):
#         super().__init__()
        
#         num_samples = straight_options.num_samples or g.dst_size
#         prev_straight = None

#         self.in_channels = in_channels = layer_options.in_channels
#         self.hidden_channels = hidden_channels = layer_options.hidden_channels
#         self.num_layers = num_layers = layer_options.num_layers
        
#         # default out_channels is hidden_channels
#         self.out_channels = out_channels = hidden_channels \
#             if layer_options.out_channels is None else layer_options.out_channels
#         self.gather_beta = gather_beta = layer_options.gather_beta
            
#         self.layer_options = layer_options
#         self.input_options = input_options
#         self.straight_options = straight_options
        
#         # initialize input layer
#         if input_options.weight:
#             self.lin_x = nn.Linear(in_channels, hidden_channels, bias=input_options.bias)
#             in_channels = hidden_channels
            
#             if straight_options.enabled and input_options.straight_enabled and not input_options.gather_first:
#                 sns = input_options.straight_num_samples or g.dst_size
#                 self.straight_x = Straight(g.dst_size, sns, beta=straight_options.beta, prev=prev_straight)
#                 prev_straight = [self.straight_x]
            
#         if input_options.gather:
#             self.gather_x = Gather(g.src_size, in_channels, beta=gather_beta)
        
#         if input_options.act is not None:
#             self.act_x = activation_resolver(
#                 input_options.act, **(input_options.act_kwargs or {}))
        
#         if input_options.norm is not None:
#             self.norm_x = normalization_resolver(
#                     input_options.norm, in_channels, **(input_options.norm_kwargs or {}))
        
#         # initialize activation layers
#         if layer_options.act is not None:
#             self.acts = nn.ModuleList()
#             for _ in range(num_layers - 1):
#                 self.acts.append(activation_resolver(
#                     layer_options.act, **(layer_options.act_kwargs or {})))
#             if layer_options.jk_mode is not None:
#                 self.acts.append(activation_resolver(
#                     layer_options.act, **(layer_options.act_kwargs or {})))
        
#         # initialize normalization layers
#         if layer_options.norm is not None:
#             self.norms = nn.ModuleList()
#             for _ in range(num_layers - 1):
#                 self.norms.append(normalization_resolver(
#                     layer_options.norm, hidden_channels, **(layer_options.norm_kwargs or {})))
#             if layer_options.jk_mode is not None:
#                 self.norms.append(normalization_resolver(
#                     layer_options.norm, hidden_channels, **(layer_options.norm_kwargs or {})))
        
#         # initialize straight layers
#         if straight_options.enabled:
#             self.straights = nn.ModuleList()
#             for _ in range(num_layers):
#                 prev_straight = [Straight(g.dst_size, num_samples, beta=straight_options.beta, prev=prev_straight)]
#                 self.straights.append(prev_straight[0])
#             # if layer_options.jk_mode is not None:
#             #     prev_straight = [Straight(g.dst_size, num_samples, beta=straight_options.beta, prev=prev_straight)]
#             #     self.straights.append(prev_straight[0])
        
#         # initialize gather and conv layers
#         self.convs = nn.ModuleList()
#         self.gathers = nn.ModuleList()
#         for _ in range(num_layers - 1):
#             self.convs.append(
#                 self.init_conv(in_channels, hidden_channels, **kwargs))
#             self.gathers.append(Gather(g.src_size, hidden_channels, beta=gather_beta))
#             in_channels = hidden_channels
        
#         if layer_options.jk_mode is None:
#             self.convs.append(
#                 self.init_conv(in_channels, out_channels, **kwargs))
#             self.gathers.append(Gather(g.dst_size, out_channels)) # only fuse embeddings
#         else:
#             self.convs.append(
#                 self.init_conv(in_channels, hidden_channels, **kwargs))
#             self.gathers.append(Gather(g.dst_size, hidden_channels, beta=gather_beta)) # only fuse embeddings
            
#             if layer_options.jk_mode != "last":
#                 self.jk = JumpingKnowledge(layer_options.jk_mode, hidden_channels, num_layers)
        
#             if layer_options.jk_mode == "cat":
#                 jk_channels = num_layers * hidden_channels
#             else:
#                 jk_channels = hidden_channels
#             self.lin_jk = nn.Linear(jk_channels, out_channels)
        
#         self.reset_parameters()

#     def init_conv(self,
#         in_channels: int,
#         out_channels: int,
#         **kwargs
#     ) -> nn.Module:
#         raise NotImplementedError

#     def reset_parameters(self):
#         if hasattr(self, "lin_x"):
#             self.lin_x.reset_parameters()
        
#         if hasattr(self, "straight_x"):
#             self.straight_x.reset_parameters()
        
#         if hasattr(self, "gather_x"):
#             self.gather_x.reset_parameters()
        
#         if hasattr(self, "norm_x"):
#             self.norm_x.reset_parameters()
        
#         if hasattr(self, "norms"):
#             for norm in self.norms:
#                 norm.reset_parameters()
        
#         if hasattr(self, "straights"):
#             for straight in self.straights:
#                 straight.reset_parameters()
        
#         for conv in self.convs:
#             conv.reset_parameters()
        
#         for gather in self.gathers:
#             gather.reset_parameters()

#         if hasattr(self, "jk"):
#             self.jk.reset_parameters()
            
#         if hasattr(self, "lin_jk"):
#             self.lin_jk.reset_parameters()
            
#     def forward(
#         self,
#         g: DistGraph,
#     ):
#         from ..utils.printer import main_print, sync_print
        
#         x = g.ndata["x"]
#         async_op = g.args.get("async_op", False)
        
#         # input layer
#         if hasattr(self, "straight_x"):
#             dst_idx, _ = self.straight_x.pop_next_shrink_helper()
#         else:
#             dst_idx = None

#         if dst_idx is not None:
#             x = x[dst_idx]

#         if hasattr(self, "gather_x") and self.input_options.gather_first:
#             x, _ = self.gather_x(x, dst_idx, g.route, async_op=async_op)
            
#         if self.input_options.dropout != 0.0:
#             x = F.dropout(x, p=self.input_options.dropout, training=self.training)
        
#         if hasattr(self, "lin_x"):
#             x = self.lin_x(x)
        
#         if hasattr(self, "act_x") and self.input_options.act_first:
#             x = self.act_x(x)
#         if hasattr(self, "norm_x"):
#             x = self.norm_x(x)
#         if hasattr(self, "act_x") and not self.input_options.act_first:
#             x = self.act_x(x)
        
#         # straight sampler
#         if hasattr(self, "straight_x"):
#             # sync_print(f"{x.size()} - {'' if dst_idx is None else dst_idx.size()}")
#             x = self.straight_x(x, dst_idx, g)
        
#         # gather features
#         if hasattr(self, "gather_x") and not self.input_options.gather_first:
#             x, _ = self.gather_x(x, dst_idx, g.route, async_op=async_op)
            
#         # conv layers
#         xs: List[Tensor] = []
#         for i in range(self.num_layers):
#             if self.layer_options.dropout != 0.0:
#                 x = F.dropout(x, p=self.layer_options.dropout, training=self.training)
            
#             if hasattr(self, "straights"):
#                 straight: Straight = self.straights[i]
#                 dst_idx, sh = straight.pop_next_shrink_helper()
            
#             with g.scoped_manager():
#                 g.ndata["x"] = x
#                 x = self.convs[i](g, sh=sh, dst_idx=dst_idx)
            
#             if i == self.num_layers - 1 and not hasattr(self, "jk"):
#                 x = self.gathers[i].fuse_embeddings(x, dst_idx, inplace=True)
#                 break
#             if hasattr(self, "acts") and self.layer_options.act_first:
#                 x = self.acts[i](x)
#             if hasattr(self, "norms"):
#                 x = self.norms[i](x)
#             if hasattr(self, "acts") and not self.layer_options.act_first:
#                 x = self.acts[i](x)
            
#             if hasattr(self, "straights"):
#                 x = self.straights[i](x, dst_idx, g)
            
#             if i == self.num_layers - 1:
#                 x = self.gathers[i].fuse_embeddings(x, dst_idx, inplace=True)
#             else:
#                 x, _ = self.gathers[i](x, dst_idx, g.route, async_op=async_op)
                
#             if hasattr(self, "jk"):
#                 xs.append(x[:g.dst_size])

#         x = self.jk(xs) if hasattr(self, "jk") else x
#         x = self.lin_jk(x) if hasattr(self, "lin_jk") else x
        
#         # sync_print(f"out: {x.size()}")
#         return x
