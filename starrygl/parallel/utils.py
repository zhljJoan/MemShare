import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor
from typing import *

from collections import defaultdict


__all__ = [
    "all_reduce_gradients",
    "all_reduce_buffers",
]

# def all_reduce_gradients(net: nn.Module, op = dist.ReduceOp.SUM, group = None, async_op: bool = False):
#     works = []
#     for p in net.parameters():
#         if p.grad is None:
#             p.grad = torch.zeros_like(p.data)
#         w = dist.all_reduce(p.grad, op=op, group=group, async_op=async_op)
#         works.append(w)
#     if async_op:
#         return works

# def all_reduce_buffers(net: nn.Module, op = dist.ReduceOp.AVG, group = None, async_op: bool = False):
#     works = []
#     for b in net.buffers():
#         w = dist.all_reduce(b.data, op=op, group=group, async_op=async_op)
#         works.append(w)
#     if async_op:
#         return works


def all_reduce_gradients(net: nn.Module, op = dist.ReduceOp.SUM, group = None, async_op: bool = False):
    device = None
    works = []

    if op is None:
        return works

    typed_numel = defaultdict(lambda: 0)
    for p in net.parameters():
        typed_numel[p.dtype] += p.numel()
        device = p.device
    
    if device is None:
        return works
    
    typed_tensors: Dict[torch.dtype, Tensor] = {}
    for t, n in typed_numel.items():
        typed_tensors[t] = torch.zeros(n, dtype=t, device=device)
    
    typed_offset = defaultdict(lambda: 0)
    for p in net.parameters():
        s = typed_offset[p.dtype]
        t = s + p.numel()
        typed_offset[p.dtype] = t

        if p.grad is not None:
            typed_tensors[p.dtype][s:t] = p.grad.flatten()
        storage = typed_tensors[p.dtype].untyped_storage()

        g = torch.empty(0, dtype=p.dtype, device=device)
        p.grad = g.set_(storage, s, p.size(), default_stride(*p.size()))
    
    for t in typed_tensors.values():
        w = dist.all_reduce(t, op=op, group=group, async_op=async_op)
        if async_op:
            works.append(w)
    return works

def all_reduce_buffers(net: nn.Module, op = dist.ReduceOp.AVG, group = None, async_op: bool = False):
    device = None
    works = []

    if op is None:
        return works

    typed_numel = defaultdict(lambda: 0)
    for p in net.buffers():
        typed_numel[p.dtype] += p.numel()
        device = p.device
    
    if device is None:
        return works
    
    typed_tensors: Dict[torch.dtype, Tensor] = {}
    for t, n in typed_numel.items():
        typed_numel[t] = torch.zeros(n, dtype=t, device=device)
    
    typed_offset = defaultdict(lambda: 0)
    for p in net.buffers():
        s = typed_offset[p.dtype]
        t = s + p.numel()
        typed_offset[p.dtype] = t

        typed_tensors[p.dtype][s:t] = p.flatten()
        storage = typed_tensors[p.dtype].untyped_storage()

        p.set_(storage, s, p.size(), default_stride(*p.size()))
    
    for t in typed_tensors.values():
        w = dist.all_reduce(t, op=op, group=group, async_op=async_op)
        if async_op:
            works.append(w)
    return works

def default_stride(*size: int) -> Tuple[int,...]:
    dims = len(size)
    stride = [1] * dims
    for i in range(1, dims):
        k = dims - i
        stride[k - 1] = stride[k] * size[k]
    return tuple(stride)
