import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.futures import Future

from torch import Tensor
from typing import *

from starrygl.parallel import all_gather_remote_objects
from .route import Route

from threading import Lock



class TensorBuffer(nn.Module):
    def __init__(self,
        channels: int,
        num_nodes: int,
        route: Route,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.num_nodes = num_nodes
        self.route = route

        self.local_lock = Lock()
        self.register_buffer("_data", torch.zeros(num_nodes, channels), persistent=False)
        self.rrefs = all_gather_remote_objects(self)
    
    @property
    def data(self) -> Tensor:
        return self.get_buffer("_data")
    
    @property
    def device(self) -> torch.device:
        return self.data.device
    
    def local_get(self, index: Optional[Tensor] = None, lock: bool = True) -> Tensor:
        if lock:
            with self.local_lock:
                return self.local_get(index, lock=False)
        if index is None:
            return self.data
        else:
            return self.data[index]

    def local_set(self, value: Tensor, index: Optional[Tensor] = None, lock: bool = True):
        if lock:
            with self.local_lock:
                return self.local_set(value, index, lock=False)
        if index is None:
            self.data.copy_(value)
        else:
            # value = value.to(self.device)
            self.data[index] = value

    def local_add(self, value: Tensor, index: Optional[Tensor] = None, lock: bool = True):
        if lock:
            with self.local_lock:
                return self.local_add(value, index, lock=False)
        if index is None:
            self.data.add_(value)
        else:
            # value = value.to(self.device)
            self.data[index] += value
    
    def local_cls(self, index: Optional[Tensor] = None, lock: bool = True):
        if lock:
            with self.local_lock:
                return self.local_cls(index, lock=False)
        if index is None:
            self.data.zero_()
        else:
            self.data[index] = 0
    
    def remote_get(self, dst: int, index: Tensor, lock: bool = True):
        return TensorBuffer._remote_call(TensorBuffer.local_get, self.rrefs[dst], index=index, lock=lock)

    def remote_set(self, dst: int, value: Tensor, index: Tensor, lock: bool = True):
        return TensorBuffer._remote_call(TensorBuffer.local_set, self.rrefs[dst], value, index=index, lock=lock)

    def remote_add(self, dst: int, value: Tensor, index: Tensor, lock: bool = True):
        return TensorBuffer._remote_call(TensorBuffer.local_add, self.rrefs[dst], value, index=index, lock=lock)

    def all_remote_get(self, index: Tensor, lock: bool = True):
        def cb0(idx):
            def f(x: torch.futures.Future[Tensor]):
                return x.value(), idx
            return f
        
        def cb1(buf):
            def f(xs: torch.futures.Future[List[torch.futures.Future]]) -> Tensor:
                for x in xs.value():
                    dat, idx = x.value()
                    # print(dat.size(), idx.size())
                    buf[idx] += dat
                return buf
            return f
        
        futs = []
        for i, (idx, remote_idx) in enumerate(self.route.parts_iter(index)):
            futs.append(self.remote_get(i, remote_idx, lock=lock).then(cb0(idx)))
        futs = torch.futures.collect_all(futs)

        buf = torch.zeros(index.size(0), self.channels, dtype=self.data.dtype, device=self.data.device)
        return futs.then(cb1(buf))

    def all_remote_set(self, value: Tensor, index: Tensor, lock: bool = True):
        futs = []
        for i, (idx, remote_idx) in enumerate(self.route.parts_iter(index)):
            futs.append(self.remote_set(i, value[idx], remote_idx, lock=lock))
        return torch.futures.collect_all(futs)

    def all_remote_add(self, value: Tensor, index: Tensor, lock: bool = True):
        futs = []
        for i, (idx, remote_idx) in enumerate(self.route.parts_iter(index)):
            futs.append(self.remote_add(i, value[idx], remote_idx, lock=lock))
        return torch.futures.collect_all(futs)

    def broadcast(self, barrier: bool = True):
        if barrier:
            dist.barrier()
        index = torch.arange(self.num_nodes, dtype=torch.long, device=self.data.device)
        data = self.all_remote_get(index, lock=True).wait()
        self.local_set(data, lock=True)
        if barrier:
            dist.barrier()
    
    # def remote_get(self, dst: int, i: int, index: Optional[Tensor] = None, global_index: bool = False, async_op: bool = False) -> Union[Tensor, Future]:
    #     return TensorBuffer._remote_call(async_op, TensorBuffer.local_get, self.rrefs[dst], i, index, global_index = global_index)

    # def remote_set(self, dst: int, i: int, value: Tensor, index: Optional[Tensor], global_index: bool = False, async_op: bool = False) -> Optional[Future]:
    #     return TensorBuffer._remote_call(async_op, TensorBuffer.local_set, self.rrefs[dst], i, value, index, global_index = global_index)

    # def remote_add(self, dst: int, i: int, value: Tensor, index: Optional[Tensor] = None, global_index: bool = False, async_op: bool = False) -> Optional[Future]:
    #     return TensorBuffer._remote_call(async_op, TensorBuffer.local_add, self.rrefs[dst], i, value, index, global_index = global_index)
    
    # def async_scatter_fw_set(self, i: int, value: Tensor, index: Optional[Tensor] = None) -> Tuple[Future]:
    #     futures: List[Future] = []
    #     for dst in range(self.world_size):
    #         val, ind = self.router.fw_value_index(dst, value, index)
    #         futures.append(self.remote_set(dst, i, val, ind, global_index=True, async_op=True))
    #     return tuple(futures)

    # def async_scatter_fw_add(self, i: int, value: Tensor, index: Optional[Tensor] = None) -> Tuple[Future]:
    #     futures: List[Future] = []
    #     for dst in range(self.world_size):
    #         val, ind = self.router.fw_value_index(dst, value, index)
    #         futures.append(self.remote_add(dst, i, val, ind, global_index=True, async_op=True))
    #     return tuple(futures)
    
    # def async_scatter_bw_set(self, i: int, value: Tensor, index: Optional[Tensor] = None) -> Tuple[Future]:
    #     futures: List[Future] = []
    #     for dst in range(self.world_size):
    #         val, ind = self.router.bw_value_index(dst, value, index)
    #         futures.append(self.remote_set(dst, i, val, ind, global_index=True, async_op=True))
    #     return tuple(futures)
    
    # def async_scatter_bw_add(self, i: int, value: Tensor, index: Optional[Tensor] = None) -> Tuple[Future]:
    #     futures: List[Future] = []
    #     for dst in range(self.world_size):
    #         val, ind = self.router.bw_value_index(dst, value, index)
    #         futures.append(self.remote_add(dst, i, val, ind, global_index=True, async_op=True))
    #     return tuple(futures)
    
    # def _idx_data(self, i: int) -> Tuple[int, Tensor]:
    #     assert -self.num_layers < i and i < self.num_layers
    #     i = (self.num_layers + i) % self.num_layers
    #     return i, self.get_buffer(f"data{i}")
    
    @staticmethod
    def _remote_call(method, rref: rpc.RRef, *args, **kwargs):
        args = (method, rref) + args
        return rpc.rpc_async(rref.owner(), TensorBuffer._method_call, args=args, kwargs=kwargs)
    
    @staticmethod
    def _method_call(method, rref: rpc.RRef, *args, **kwargs):
        self: TensorBuffer = rref.local_value()
        index = kwargs["index"]
        kwargs["index"] = self.route.to_local_ids(index)
        return method(self, *args, **kwargs)
    