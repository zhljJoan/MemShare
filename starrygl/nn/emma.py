import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch import Tensor
from typing import *

from torch_scatter import segment_csr, gather_csr
from torch_sparse import SparseTensor


__all__ = [
    "EmmaAttention",
    "EmmaSum",
]


class EmmaAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "his_x",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "his_m",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "inv_w",
            torch.empty(0),
            persistent=False,
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.get_buffer("his_x").zero_()
        self.get_buffer("his_m").fill_(-torch.inf)
        self.get_buffer("inv_w").zero_()
    
    def forward(self, x: Tensor, max_a: Tensor, agg_n: Tensor):
        if self.training:
            his_x = self.get_buffer("his_x")
            his_m = self.get_buffer("his_m")
            inv_w = self.get_buffer("inv_w")

            x = EmmaAttentionFunction.apply(
                x, max_a, his_x, his_m, agg_n, inv_w)
        else:
            inv_w = 1.0 / agg_n.data
            inv_w[agg_n == 0] = 0.0

            self._copy_or_clone("his_x", x)
            self._copy_or_clone("his_m", max_a)
            self._copy_or_clone("inv_w", inv_w)
        return x
    
    def _copy_or_clone(self, name: str, x: Tensor):
        _x = self.get_buffer(name)
        if _x.size() != x.size():
            self.register_buffer(
                name, x.data.clone(), persistent=False)
        else:
            _x.copy_(x.data)
    
    @staticmethod
    def softmax_gat(
        src_a: Tensor,
        dst_a: Tensor,
        adj_t: SparseTensor,
        negative_slope: float = 0.01,
    ) -> Tuple[SparseTensor, Tensor]:
        assert src_a.dim() in {1, 2}
        assert src_a.dim() == dst_a.dim()

        ptr, ind, val = adj_t.csr()
        
        a = src_a[ind] + gather_csr(dst_a, ptr)
        if val is not None:
            assert val.dim() == src_a.dim()
            a = a + val

        a = F.leaky_relu(a, negative_slope=negative_slope)
        with torch.no_grad():
            max_a = torch.full_like(dst_a, -torch.inf)
            max_a = segment_csr(a, ptr, reduce="max", out=max_a)

        exp_a = torch.exp(a - gather_csr(max_a, ptr))
        sum_exp_a = segment_csr(exp_a, ptr, reduce="sum")
        exp_a = exp_a / gather_csr(sum_exp_a, ptr)
        with torch.no_grad():
            max_a.add_(sum_exp_a.log())

        adj_t = SparseTensor(rowptr=ptr, col=ind, value=exp_a)
        return adj_t, max_a
    
    @staticmethod
    def apply_gat(
        x: Tensor,
        src_a: Tensor,
        dst_a: Tensor,
        adj_t: SparseTensor,
        negative_slope: float = 0.01,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        adj_t, max_a = EmmaAttention.softmax_gat(
            src_a=src_a, dst_a=dst_a,
            adj_t=adj_t, negative_slope=negative_slope,
        )

        ptr, ind, val = adj_t.csr()
        if val.dim() == 1:
            assert x.dim() == 2
            x = adj_t @ x
        elif val.dim() == 2:
            assert x.dim() == 3
            assert x.size(1) == val.size(1)
            xs = []
            for i in range(x.size(1)):
                xs.append(
                    SparseTensor(
                        rowptr=ptr, col=ind, value=val[:,i],
                    ) @ x[:,i,:]
                )
            x = torch.cat(xs, dim=1).view(-1, *x.shape[1:])

        with torch.no_grad():
            agg_n = torch.ones_like(ind)
            agg_n = segment_csr(agg_n, ptr, reduce="sum")
        return x, max_a, agg_n

class EmmaAttentionFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        x: Tensor,
        max_a: Tensor,
        his_x: Tensor,
        his_m: Tensor,
        agg_n: Tensor,
        inv_w: Tensor,
    ):
        assert x.dim() in {2, 3}
        assert x.dim() == his_x.dim()
        assert max_a.dim() == his_m.dim()

        beta = (1.0 - inv_w * agg_n).clamp_(0.0, 1.0)
        if x.dim() == 2:
            assert max_a.dim() == 1
        elif x.dim() == 3:
            assert max_a.dim() == 2
            beta = beta.unsqueeze_(-1)

        max_m = torch.max(max_a, his_m)

        p = (his_m - max_m).nan_to_num_(0.0).exp_().mul_(beta)
        q = (max_a - max_m).nan_to_num_(0.0).exp_()

        t = p + q
        p.div_(t).unsqueeze_(-1)
        q.div_(t).unsqueeze_(-1)

        his_x.mul_(p).add_(x * q)
        his_m.copy_(max_m).add_(t.log_())

        ctx.save_for_backward(q)
        return his_x
    
    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        q, = ctx.saved_tensors
        return grad * q, None, None, None, None, None

class EmmaSum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "his_x",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "inv_w",
            torch.empty(0),
            persistent=False,
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.get_buffer("his_x").zero_()
        self.get_buffer("inv_w").zero_()
    
    def forward(self, x: Tensor, agg_n: Tensor, aggr: str = "sum"):
        assert aggr in {"sum", "mean"}

        if self.training:
            his_x = self.get_buffer("his_x")
            inv_w = self.get_buffer("inv_w")

            x = EmmaSumFunction.apply(x, his_x, agg_n, inv_w)
        else:
            inv_w = 1.0 / agg_n.data
            inv_w[agg_n == 0] = 0.0

            self._copy_or_clone("his_x", x)
            self._copy_or_clone("inv_w", inv_w)

        if aggr == "mean":
            x = x * inv_w[:,None]
        return x
    
    def _copy_or_clone(self, name: str, x: Tensor):
        _x = self.get_buffer(name)
        if _x.size() != x.size():
            self.register_buffer(
                name, x.data.clone(), persistent=False)
        else:
            _x.copy_(x.data)

class EmmaSumFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        x: Tensor,
        his_x: Tensor,
        agg_n: Tensor,
        inv_w: Tensor,
    ):
        assert x.dim() == 2
        assert his_x.dim() == x.dim()

        beta = (1.0 - inv_w * agg_n) \
            .clamp_(0.0, 1.0).unsqueeze_(-1)

        his_x.mul_(beta).add_(x)
        
        # ctx.save_for_backward(inv_w)
        return his_x
    
    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        # inv_w, = ctx.saved_tensors
        # return grad * inv_w[:,None], None, None, None
        return grad, None, None, None