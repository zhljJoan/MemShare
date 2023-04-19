import torch
import torch.nn as nn
import torch.autograd as autograd

from torch import Tensor
from contextlib import contextmanager
from typing import *


class StraightContext:
    def __init__(self, this) -> None:
        self.this = this

class Straight(nn.Module):
    def __init__(self,
        num_nodes: int,
        p: int = 2,
        dim: int = -1,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.norm_p = p
        self.norm_dim = dim
        self.norm_beta = beta
        self.register_buffer("grad_norm", torch.ones(num_nodes))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        grad_norm = self.get_buffer("grad_norm")
        nn.init.constant_(grad_norm, 1.0)
    
    def forward(self, values: Tensor, active: Optional[Tensor] = None) -> Tensor:
        with self._manager(self):
            return StraightFunction.apply(values, active)
    
    def multinomial(self,
        num_samples: int,
        replacement: bool = False,
    ) -> Tensor:
        w = self.get_buffer("grad_norm")
        if num_samples <= 0:
            return torch.arange(self.num_nodes, dtype=torch.long, device=w.device)
        
        # print(w)
        w = w / w.sum()
        return torch.multinomial(w, num_samples=num_samples, replacement=replacement)
    
    def multinomial_mask(self,
        num_samples: int,
        replacement: bool = False,
    ) -> Tensor:
        device = self.get_buffer("grad_norm").device
        if num_samples <= 0:
            return torch.ones(self.num_nodes, dtype=torch.bool, device=device)
        
        w = self.multinomial(num_samples, replacement)
        m = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
        m[w] = True
        return m
    
    @contextmanager
    def _manager(self, this):
        global _global_straight_context
        stacked = _global_straight_context

        try:
            _global_straight_context = StraightContext(this=this)
            yield _global_straight_context
        finally:
            _global_straight_context = stacked
    

class StraightFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        values: Tensor,
        active: Optional[Tensor],
    ):
        stx = _last_global_straight_context()
        this: Straight = stx.this
        if this.training:
            ctx.this = this
            ctx.save_for_backward(active)
        return values

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        this: Straight = ctx.this
        active,  = ctx.saved_tensors
        
        grad_norm = this.get_buffer("grad_norm")
        if this.norm_beta != 1.0:
            grad_norm.mul_(this.norm_beta)
        
        if active is None:
            x = grad.detach()
            x = x.norm(p=this.norm_p, dim=this.norm_dim)
            grad_norm[:] = x
        else:
            if grad.size(0) == grad_norm.size(0):
                x = grad.detach()[active]
            else:
                x = grad.detach()
            x = x.norm(p=this.norm_p, dim=this.norm_dim)
            grad_norm[active] = x
        return grad, None

#### private functions
_global_straight_context: Optional[StraightContext] = None
def _last_global_straight_context() -> StraightContext:
    global _global_straight_context
    assert _global_straight_context is not None
    return _global_straight_context


if __name__ == "__main__":
    s = Straight(3, beta=1.1)
    x = torch.rand(3, 10).requires_grad_()
    m = torch.tensor([0, 1, 0], dtype=torch.bool)

    s(x).sum().backward()
    print(s.grad_norm)
    print(s.multinomial(2))

    s(x, m).sum().backward()
    print(s.grad_norm)
    print(s.multinomial(2))
    
    s(x[m], m).sum().backward()
    print(s.grad_norm)
    print(s.multinomial(2))
    
    print(s.multinomial_mask(2))