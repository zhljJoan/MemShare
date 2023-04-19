import torch
import torch.nn as nn
import torch.autograd as autograd

from torch import Tensor
from contextlib import contextmanager
from typing import *

from .route import Route, GatherWork
from .cache import EmbeddingCache


class GatherContext:
    def __init__(self,
        this,
        route: Route,
        async_op: bool,
    ) -> None:
        self.this = this
        self.route = route
        self.async_op = async_op

class Gather(nn.Module):
    def __init__(self,
        num_nodes: int,
        num_features: Optional[int] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.beta = beta
        if num_features is None:
            self.register_buffer("last_embd", torch.zeros(num_nodes))
        else:
            self.register_buffer("last_embd", torch.zeros(num_nodes, num_features))
        self.last_fw_work: Optional[GatherWork] = None
        self.last_bw_work: Optional[GatherWork] = None
        self.reset_parameters()
    
    def forward(self,
        values: Tensor,
        active: Optional[Tensor],
        route: Route,
        async_op: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        with self._gather_manager(route=route, async_op=async_op):
            return GatherFunction.apply(values, active)
    
    def reset_parameters(self):
        last_embd = self.get_buffer("last_embd")
        nn.init.normal_(last_embd, mean=0, std=1)
    
    def fuse_embeddings(self,
        values: Tensor,
        active: Optional[Tensor] = None,
        inplace: bool = False,
    ) -> Tensor:
        last_embd = self.get_buffer("last_embd")
        return GatherFuseFunction.apply(values, active, last_embd, self.beta, inplace, self.training)
        # if not inplace:
        #     last_embd = last_embd.clone()
            
        # if active is None:
        #     if self.beta != 1.0:
        #         values = values * self.beta + last_embd * (1 - self.beta)
        #     last_embd[:] = values
        # else:
        #     if values.size(0) == active.size(0):
        #         values = values[active]
        #     if self.beta != 1.0:
        #         values = values * self.beta + last_embd[active] * (1 - self.beta)
        #     last_embd[active] = values
        # return last_embd
    
    @contextmanager
    def _gather_manager(self, route: Route, async_op: bool):
        global _global_gather_context
        stacked = _global_gather_context

        try:
            _global_gather_context = GatherContext(
                this=self,
                route=route,
                async_op=async_op,
            )
            yield _global_gather_context
        finally:
            _global_gather_context = stacked

class GatherFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        values: Tensor,
        active: Optional[Tensor] = None,
    ):
        gather_ctx = _last_global_gather_context()
        this: Gather = gather_ctx.this
        route: Route = gather_ctx.route
        async_op: bool = gather_ctx.async_op
        
        current_work = route.gather_forward(values, active, async_op=async_op)
        if async_op:
            work = this.last_fw_work or current_work
            this.last_fw_work = current_work
        else:
            work = current_work
        
        recv_values, recv_active = work.get()
        recv_values = this.fuse_embeddings(
            values=recv_values,
            active=recv_active,
            inplace=True,
        )
        
        if this.training:
            # 如果输入的values是收缩过的，求解梯度的时候需要去除空洞
            if active is not None and values.size(0) < active.size(0):
                ctx.shrink_grad = True
                ctx.save_for_backward(active, recv_active)
            else:
                ctx.shrink_grad = False
                ctx.save_for_backward(recv_active)
            ctx.this = this
            ctx.route = route
            ctx.async_op = async_op
        return recv_values, recv_active

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad_values: Tensor,
        grad_active: Optional[Tensor],
    ):
        this: Gather = ctx.this
        route: Route = ctx.route
        async_op: bool = ctx.async_op
        shrink_grad: bool = ctx.shrink_grad
        
        with torch.no_grad():
            # # 反向传播激活值是沿着前向传播的反方向进行
            if shrink_grad:
                recv_active, grad_active = ctx.saved_tensors
            else:
                grad_active, = ctx.saved_tensors

            current_work = route.gather_backward(grad_values, grad_active, async_op=async_op)
            if async_op:
                work = this.last_bw_work or current_work
                this.last_bw_work = current_work
            else:
                work = current_work
            
            recv_values = work.get_values()
            
            if shrink_grad:
                recv_values = recv_values[recv_active]
            return recv_values, None


class GatherFuseFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        values: Tensor,
        active: Optional[Tensor],
        last_embd: Tensor,
        beta: float,
        inplace: bool,
        training: bool,
    ):
        if not inplace:
            last_embd = last_embd.clone()
        ctx.beta = beta
        
        if active is None:
            if beta != 1.0:
                values = values * beta + last_embd * (1 - beta)
            last_embd[:] = values
            ctx.shrink_grad = False
        else:
            if values.size(0) == active.size(0):
                values = values[active]
                ctx.shrink_grad = False
            else:
                if training:
                    ctx.save_for_backward(active)
                ctx.shrink_grad = True
                
            if beta != 1.0:
                values = values * beta + last_embd[active] * (1 - beta)
            last_embd[active] = values
        return last_embd

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        if ctx.shrink_grad:
            active, = ctx.saved_tensors
            return grad[active] * ctx.beta, None, None, None, None, None
        else:
            return grad * ctx.beta, None, None, None, None, None


#### private functions
_global_gather_context: Optional[GatherContext] = None
def _last_global_gather_context() -> GatherContext:
    global _global_gather_context
    assert _global_gather_context is not None
    return _global_gather_context
