import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.loader import BatchHandle



class BaseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        return x
    
    def update_forward(self, handle: BatchHandle, edge_index: Tensor, edge_attr: Optional[Tensor] = None):
        x = handle.fetch_feat()
        with torch.no_grad():
            x = self.forward(x, edge_index, edge_attr)
        handle.update_feat(x)
    
    def block_backward(self, handle: BatchHandle, edge_index: Tensor, edge_attr: Optional[Tensor] = None):
        x = handle.fetch_feat().requires_grad_()
        g = handle.fetch_grad()
        self.forward(x, edge_index, edge_attr).backward(g)
        handle.accumulate_grad(x.grad)
        x.grad = None
    
    def all_reduce_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    
class BaseModel(nn.Module):
    def __init__(self,
        num_features: int,
        layers: List[int],
        prev_layer: bool = False,
        post_layer: bool = False,
    ) -> None:
        super().__init__()

    def init_prev_layer(self) -> Tensor:
        pass

    def init_post_layer(self) -> Tensor:
        pass

    def init_conv_layer(self) -> Tensor:
        pass