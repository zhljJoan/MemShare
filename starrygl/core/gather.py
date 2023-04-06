import torch
import torch.nn as nn
import torch.autograd as autograd

from torch import Tensor
from contextlib import contextmanager
from typing import *

from .route import Route, GatherWork


class GatherContext:
    def __init__(self,
        this,
        route: Route,
        async_op: bool,
    ) -> None:
        self.this = this
        self.route = route
        self.async_op = async_op
    
    def gather(self,
        values: Tensor,
        active: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        values, active = GatherFunction.apply(values, active)
        return values, active

class Gather:
    def __init__(self) -> None:
        self.cache: Dict[str, Tensor] = {}
        self.last_fw_work: Optional[GatherWork] = None
        self.last_bw_work: Optional[GatherWork] = None
    
    def __call__(self,
        values: Tensor,
        active: Optional[Tensor],
        route: Route,
        async_op: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        with self._gather_manager(route=route, async_op=async_op) as gtx:
            values, active = gtx.gather(values=values, active=active)
        return values, active
    
    def clear(self):
        self.cache.clear()
        self.last_fw_work = None
        self.last_bw_work = None
    
    @contextmanager
    def _gather_manager(self, route: Route, async_op: bool):
        global _global_gather_context
        prev_gtx = _global_gather_context

        try:
            _global_gather_context = GatherContext(
                this=self, route=route, async_op=async_op)
            yield _global_gather_context
        finally:
            _global_gather_context = prev_gtx

class GatherFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        values: Tensor,
        active: Optional[Tensor] = None,
    ):
        gtx = _last_global_gather_context()
        this: Gather = gtx.this
        route: Route = gtx.route

        fw_key = "__forward_activation__"
        cached = this.cache.get(fw_key)
        if cached is None:
            cached = torch.zeros(
                size=(route.dst_size,) + values.shape[1:],
                dtype=values.dtype,
                device=values.device,
            )
        
        current_work = route.gather_forward(values, active, async_op=gtx.async_op)
        if gtx.async_op:
            work = this.last_fw_work or current_work
            this.last_fw_work = current_work
        else:
            work = current_work
        
        recv_values = work.get_values()
        recv_active = work.get_active()

        cached[recv_active] = recv_values
        this.cache[fw_key] = cached
        
        # 如果输入的values是收缩过的，求解梯度的时候需要去除空洞
        if active is not None and values.size(0) < active.size(0):
            ctx.shrink_values = True
            ctx.save_for_backward(active, recv_active)
        else:
            ctx.shrink_values = False
            ctx.save_for_backward(recv_active)
        ctx.gtx = gtx

        return cached, recv_active

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad_values: Tensor,
        grad_active: Optional[Tensor], # always None
    ):
        gtx: GatherContext = ctx.gtx
        this: Gather = gtx.this
        route: Route = gtx.route

        # 返回的梯度是否需要去空洞
        shrink_values: bool = ctx.shrink_values
        with torch.no_grad():
            # # 反向传播激活值是沿着前向传播的反方向进行
            if shrink_values:
                active, recv_active = ctx.saved_tensors
            else:
                recv_active, = ctx.saved_tensors
            
            bw_key = "__backward_gradient__"
            cached = this.cache.get(bw_key)
            if cached is None:
                cached = torch.zeros(
                    size=(route.src_size,) + grad_values.shape[1:],
                    dtype=grad_values.dtype,
                    device=grad_values.device,
                )
            
            current_work = route.gather_backward(grad_values, recv_active, async_op=gtx.async_op)
            if gtx.async_op:
                work = this.last_bw_work or current_work
                this.last_bw_work = current_work
            else:
                work = current_work
            
            recv_grad_values = work.get_values()
            recv_grad_active = work.get_active()
            
            cached[recv_grad_active] = recv_grad_values
            this.cache[bw_key] = cached

            if shrink_values:
                return cached[active], None
            else:
                return cached, None


#### private functions
_global_gather_context: Optional[GatherContext] = None
def _last_global_gather_context() -> GatherContext:
    global _global_gather_context
    assert _global_gather_context is not None
    return _global_gather_context
