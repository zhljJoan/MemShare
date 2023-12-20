import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor
from typing import *


__all__ = [
    "all_reduce_gradients",
    "all_reduce_buffers",
]

def all_reduce_gradients(net: nn.Module, op = dist.ReduceOp.SUM, group = None):
    for p in net.parameters():
        dist.all_reduce(p.grad, op=op, group=group)

def all_reduce_buffers(net: nn.Module, op = dist.ReduceOp.AVG, group = None):
    for b in net.buffers():
        dist.all_reduce(b.data, op=op, group=group)
