import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributed as dist

from torch import Tensor
from typing import *


__all__ = [
    "SyncBatchNorm",
]

class SyncBatchNorm(nn.Module):
    def __init__(self,
        num_features: Union[int, torch.Size],
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
    
    def forward(self, x: Tensor) -> Tensor:
        return SyncBatchNormFunction.apply(
            x,
            self.running_mean, self.running_var,
            self.weight, self.bias,
            self.training, self.momentum, self.eps,
        )
    
    def reset_parameters(self):
        self.running_mean.data.fill_(0.0)
        self.running_var.data.fill_(1.0)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
    
    @classmethod
    def convert_sync_batchnorm(cls, net: nn.Module) -> nn.Module:
        if isinstance(net, nn.modules.batchnorm._BatchNorm):
            new_net = SyncBatchNorm(
                num_features=net.num_features,
                eps=net.eps,
                momentum=net.momentum,
            ).to(net.weight.device)
            new_net.weight.data[:] = net.weight.data
            new_net.bias.data[:] = net.bias.data
            new_net.get_buffer("running_mean").data[:] = net.get_buffer("running_mean").data
            new_net.get_buffer("running_var").data[:] = net.get_buffer("running_var").data
            return new_net
        else:
            for name, child in list(net.named_children()):
                net.add_module(name, cls.convert_sync_batchnorm(child))
            return net

class SyncBatchNormFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        x: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        training: bool,
        momentum: float,
        eps: float,
    ):
        if not training:
            mean = running_mean
            var = running_var
        else:
            ws = torch.zeros(1, dtype=x.dtype, device=x.device) + x.size(0)
            ws_req = dist.all_reduce(ws, op=dist.ReduceOp.SUM, async_op=True)

            if x.size(0) > 0:
                sum_x = x.sum(dim=0)
            else:
                sum_x = torch.zeros(
                    size=(1,) + x.shape[1:],
                    dtype=x.dtype,
                    device=x.device,
                )
            sum_x_req = dist.all_reduce(sum_x, op=dist.ReduceOp.SUM, async_op=True)

            if x.size(0) > 0:
                sum_x2 = (x**2).sum(dim=0)
            else:
                sum_x2 = torch.zeros(
                    size=(1,) + x.shape[1:],
                    dtype=x.dtype,
                    device=x.device,
                )
            sum_x2_req = dist.all_reduce(sum_x2, op=dist.ReduceOp.SUM, async_op=True)

            ws_req.wait()
            whole_size = ws.item()

            sum_x_req.wait()
            mean = sum_x / whole_size

            sum_x2_req.wait()
            # var = sum_x2 / whole_size - mean ** 2
            var = (sum_x2 - mean * sum_x) / whole_size

            running_mean.mul_(1 - momentum).add_(mean * momentum)
            running_var.mul_(1 - momentum).add_(var * momentum)

        std = torch.sqrt(var + eps)
        x_hat = (x - mean) / std
        if training:
            ctx.save_for_backward(x_hat, weight, std)
            ctx.whole_size = whole_size
        return x_hat * weight + bias

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        x_hat, weight, std = ctx.saved_tensors
        dbias = grad.sum(dim=0)
        dbias_req = dist.all_reduce(dbias, op=dist.ReduceOp.SUM, async_op=True)
        
        dweight = (grad * x_hat).sum(dim=0)
        dweight_req = dist.all_reduce(dweight, op=dist.ReduceOp.SUM, async_op=True)

        dbias_req.wait()
        dweight_req.wait()

        n = ctx.whole_size
        dx = (weight / n) / std * (n * grad - dbias - x_hat * dweight)
        return dx, None, None, dweight, dbias, None, None, None
