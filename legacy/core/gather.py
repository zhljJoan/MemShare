# import torch
# import torch.nn as nn
# import torch.autograd as autograd

# from torch import Tensor
# from contextlib import contextmanager
# from typing import *

# from .route import Route, GatherWork
# # from .cache import CachedEmbeddings


# class GatherContext:
#     def __init__(self,
#         this,
#         route: Route,
#         async_op: bool,
#     ) -> None:
#         self.this = this
#         self.route = route
#         self.async_op = async_op

# class Gather(nn.Module):
#     def __init__(self,
#         num_nodes: int,
#         num_features: Optional[int] = None,
#         beta: float = 1.0,
#     ) -> None:
#         super().__init__()
#         self.beta = beta
#         if num_features is None:
#             self.register_buffer("last_embd", torch.zeros(num_nodes))
#         else:
#             self.register_buffer("last_embd", torch.zeros(num_nodes, num_features))
        
#         self.last_fw_work: Optional[GatherWork] = None
#         self.last_bw_work: Optional[GatherWork] = None
#         self.reset_parameters()
    
#     def forward(self,
#         val: Tensor,
#         idx: Optional[Tensor],
#         route: Route,
#         async_op: bool = False,
#     ) -> Tuple[Tensor, Tensor]:
#         with self._manager(route=route, async_op=async_op):
#             return GatherFunction.apply(val, idx)
    
#     def reset_parameters(self):
#         last_embd = self.get_buffer("last_embd")
#         nn.init.normal_(last_embd, mean=0, std=1)
    
#     def fuse_embeddings(self,
#         val: Tensor,
#         idx: Optional[Tensor] = None,
#         inplace: bool = False,
#     ) -> Tensor:
#         last_embd = self.get_buffer("last_embd")
#         return GatherFuseFunction.apply(val, idx, last_embd, self.beta, inplace, self.training)
    
#     @contextmanager
#     def _manager(self, route: Route, async_op: bool):
#         global _global_gather_context
#         stacked = _global_gather_context

#         try:
#             _global_gather_context = GatherContext(
#                 this=self,
#                 route=route,
#                 async_op=async_op,
#             )
#             yield _global_gather_context
#         finally:
#             _global_gather_context = stacked

# class GatherFunction(autograd.Function):
#     @staticmethod
#     def forward(
#         ctx: autograd.function.FunctionCtx,
#         val: Tensor,
#         idx: Optional[Tensor] = None,
#     ):
#         gather_ctx = _last_global_gather_context()
#         this: Gather = gather_ctx.this
#         route: Route = gather_ctx.route
#         async_op: bool = gather_ctx.async_op
#         return_idx: bool = idx is not None
        
#         current_work = route.gather_forward(val, idx, async_op=async_op, return_idx=return_idx)
#         if async_op:
#             work = this.last_fw_work or current_work
#             this.last_fw_work = current_work
#         else:
#             work = current_work
        
#         recv_val, recv_idx = work.get()
#         recv_val = recv_val if recv_idx is None else recv_val[recv_idx]
#         recv_val = this.fuse_embeddings(recv_val, recv_idx, inplace=True)
        
#         if this.training:
#             ctx.save_for_backward(idx, recv_idx)
#             ctx.this = this
#             ctx.route = route
#             ctx.async_op = async_op
#         return recv_val, recv_idx

#     @staticmethod
#     def backward(
#         ctx: autograd.function.FunctionCtx,
#         val_grad: Tensor,
#         idx_grad: Optional[Tensor],
#     ):
#         this: Gather = ctx.this
#         route: Route = ctx.route
#         async_op: bool = ctx.async_op
        
#         with torch.no_grad():
#             recv_idx, idx_grad = ctx.saved_tensors
#             if idx_grad is not None:
#                 val_grad = val_grad[idx_grad]

#             current_work = route.gather_backward(val_grad, idx_grad, async_op=async_op, return_idx=False)
#             if async_op:
#                 work = this.last_bw_work or current_work
#                 this.last_bw_work = current_work
#             else:
#                 work = current_work
            
#             recv_val = work.get_val()
#             if recv_idx is not None:
#                 recv_val = recv_val[recv_idx]
#             return recv_val, None

# class GatherFuseFunction(autograd.Function):
#     @staticmethod
#     def forward(
#         ctx: autograd.function.FunctionCtx,
#         val: Tensor,
#         idx: Optional[Tensor],
#         last_embd: Tensor,
#         beta: float,
#         inplace: bool,
#         training: bool,
#     ):
#         if not inplace:
#             last_embd = last_embd.clone()
#         ctx.beta = beta
        
#         if idx is None:
#             assert val.size(0) == last_embd.size(0)
#             if beta != 1.0:
#                 last_embd.mul_(1 - beta).add_(val * beta)
#             else:
#                 last_embd[:] = (val)
#         else:
#             assert val.size(0) == idx.size(0)
#             if beta != 1.0:
#                 last_embd[idx] = last_embd[idx] * (1 - beta) + val * beta
#             else:
#                 last_embd[idx] = val
        
#         if training:
#             ctx.beta = beta
#             ctx.save_for_backward(idx)
#         return last_embd

#     @staticmethod
#     def backward(
#         ctx: autograd.function.FunctionCtx,
#         grad: Tensor,
#     ):
#         beta: float = ctx.beta
#         idx, = ctx.saved_tensors
        
#         if idx is not None:
#             grad = grad[idx]
        
#         if beta != 1.0:
#             grad = grad * beta
            
#         return grad, None, None, None, None, None

# #### private functions
# _global_gather_context: Optional[GatherContext] = None
# def _last_global_gather_context() -> GatherContext:
#     global _global_gather_context
#     assert _global_gather_context is not None
#     return _global_gather_context
