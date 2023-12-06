import torch
import torch.autograd as autograd

from torch import Tensor
from typing import *

class AFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        a: Tensor,
        b: Tensor,
    ):
        return a, b.float()

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        a: Tensor,
        b: Tensor,
    ):
        print(a, b)
        return a, b.bool()

if __name__ == "__main__":
    x = torch.rand(10).requires_grad_()
    y = torch.ones(10).bool()
    a, b = AFunction.apply(x, y)
    (a + b).sum().backward()
    print(a, b)
    print(x.grad, y.grad)
