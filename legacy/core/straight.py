# import torch
# import torch.nn as nn
# import torch.autograd as autograd

# from torch import Tensor
# from contextlib import contextmanager
# from typing import *



# class StraightContext:
#     def __init__(self, this, g) -> None:
#         self.this = this
#         self.g = g

# class Straight(nn.Module):
#     def __init__(self,
#         num_nodes: int,
#         num_samples: int,
#         norm_kwargs: Optional[Dict[str, Any]] = None,
#         beta: float = 1.0,
#         prev: Optional[List[Any]] = None,
#     ) -> None:
#         super().__init__()
#         assert num_samples <= num_nodes
#         self.num_nodes = num_nodes
#         self.num_samples = num_samples
#         self.norm_kwargs = norm_kwargs or dict(p=2, dim=-1)
#         self.beta = beta
#         self.prev = prev
        
#         self.register_buffer("last_w", torch.ones(num_nodes))

#         self._next_idx = None
#         self._next_shrink_helper = None

#         self.reset_parameters()
    
#     def reset_parameters(self):
#         last_w = self.get_buffer("last_w")
#         nn.init.constant_(last_w, 1.0)
    
#     def forward(self,
#         val: Tensor,
#         idx: Optional[Tensor],
#         g,
#     ) -> Tensor:
#         with self._manager(self, g):
#             return StraightFunction.apply(val, idx)
    
#     def pop_next_shrink_helper(self) -> Tuple[Optional[Tensor], Any]:
#         if not self.training:
#             return None, None
        
#         next_idx = self._next_idx
#         self._next_idx = None

#         next_sh = self._next_shrink_helper
#         self._next_shrink_helper = None
        
#         return next_idx, next_sh
    
#     def _sample_next(self) -> Tensor:
#         w = self.get_buffer("last_w")
#         if self._next_idx is None:
#             if self.num_samples < w.size(0):
#                 self._next_idx = self.sample_impl(w)
#             else:
#                 self._next_idx = torch.arange(w.size(0), dtype=torch.long, device=w.device)
#         elif self.num_samples < self._next_idx.size(0):
#             idx = self.sample_impl(w[self._next_idx])
#             self._next_idx = self._next_idx[idx]
#         return self._next_idx
    
#     def sample_impl(self, w: Tensor) -> Tensor:
#         w = w / w.sum()
#         return torch.multinomial(w, num_samples=self.num_samples, replacement=False)

#     # def multinomial(self,
#     #     num_samples: int,
#     #     replacement: bool = False,
#     # ) -> Tensor:
#     #     w = self.get_buffer("last_w")
#     #     if num_samples <= 0:
#     #         return torch.arange(self.num_nodes, dtype=torch.long, device=w.device)
        
#     #     w = w / w.sum()
#     #     return torch.multinomial(w, num_samples=num_samples, replacement=replacement)
    
#     # def multinomial_mask(self,
#     #     num_samples: int,
#     #     replacement: bool = False,
#     # ) -> Tensor:
#     #     w = self.get_buffer("last_w")
#     #     if num_samples <= 0:
#     #         return torch.ones(self.num_nodes, dtype=torch.bool, device=w.device)
        
#     #     w = self.multinomial(num_samples, replacement)
#     #     m = torch.zeros(self.num_nodes, dtype=torch.bool, device=w.device)
#     #     m[w] = True
#     #     return m
    
#     @contextmanager
#     def _manager(self, this, g):
#         global _global_straight_context
#         stacked = _global_straight_context

#         try:
#             _global_straight_context = StraightContext(this=this, g=g)
#             yield _global_straight_context
#         finally:
#             _global_straight_context = stacked
    

# class StraightFunction(autograd.Function):
#     @staticmethod
#     def forward(
#         ctx: autograd.function.FunctionCtx,
#         val: Tensor,
#         idx: Optional[Tensor],
#     ):
#         from ..graph import DistGraph

#         stx = _last_global_straight_context()
#         this: Straight = stx.this
#         g: DistGraph = stx.g
        
#         last_w = this.get_buffer("last_w")
#         if idx is None:
#             assert val.size(0) == last_w.size(0)
#         else:
#             assert val.size(0) == idx.size(0)
            
#         if this.training:
#             ctx.this = this
#             ctx.g = g
#             ctx.save_for_backward(idx)
#         return val

#     @staticmethod
#     def backward(
#         ctx: autograd.function.FunctionCtx,
#         grad: Tensor,
#     ):
#         from ..graph import DistGraph

#         this: Straight = ctx.this
#         g: DistGraph = ctx.g
#         idx,  = ctx.saved_tensors
        
#         last_w = this.get_buffer("last_w")
#         if this.beta != 1.0:
#             last_w.mul_(this.beta)
            
#         norm = grad.norm(**this.norm_kwargs)
#         if idx is None:
#             last_w[:] = norm
#         else:
#             last_w[idx] = norm
        
#         if this.prev is not None or this._next_idx is None:
#             from ..nn.convs.utils import ShrinkHelper
#             dst_idx = this._sample_next()
        
#         if this.prev is not None and this._next_idx is not None:
#             this._next_shrink_helper = ShrinkHelper(g, dst_idx)
            
#             src_idx = this._next_shrink_helper.src_idx
#             work = g.route.gather_backward(
#                 src_idx, src_idx, async_op=False, return_idx=True)

#             prev_dst_idx = work.get_idx()
#             for p in this.prev:
#                 assert isinstance(p, Straight)
#                 p._next_idx = prev_dst_idx
#         return grad, None

# #### private functions
# _global_straight_context: Optional[StraightContext] = None
# def _last_global_straight_context() -> StraightContext:
#     global _global_straight_context
#     assert _global_straight_context is not None
#     return _global_straight_context


# if __name__ == "__main__":
#     s = Straight(3, beta=1.1)
#     x = torch.rand(3, 10).requires_grad_()
#     m = torch.tensor([0, 1, 0], dtype=torch.bool)

#     s(x).sum().backward()
#     print(s.grad_norm)
#     print(s.multinomial(2))

#     s(x, m).sum().backward()
#     print(s.grad_norm)
#     print(s.multinomial(2))
    
#     s(x[m], m).sum().backward()
#     print(s.grad_norm)
#     print(s.multinomial(2))
    
#     print(s.multinomial_mask(2))