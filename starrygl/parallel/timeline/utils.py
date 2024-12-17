import torch

from torch import Tensor
from typing import *


def vector_backward(
    vec_loss: Sequence[Tensor],
    vec_grad: Sequence[Tensor],
):
    loss: List[Tensor] = []
    grad: List[Tensor] = []
    for x, g in zip(vec_loss, vec_grad):
        if g is None:
            continue
        if not x.requires_grad:
            continue
        loss.append(x.flatten())
        grad.append(g.flatten())
        
    if loss:
        loss = torch.cat(loss, dim=0)
        grad = torch.cat(grad, dim=0)
        loss.backward(grad)