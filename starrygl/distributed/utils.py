import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from torch import Tensor
from torch.types import Number
from typing import *

from starrygl.distributed.context import DistributedContext
from torch_sparse import SparseTensor
from .cclib import all_to_all_s


class TensorAccessor:
    def __init__(self, data: Tensor) -> None:
        from .context import DistributedContext

        self._data = data
        self._ctx = DistributedContext.get_default_context()
        if self._ctx._use_rpc is True:
            self._rref = rpc.RRef(data)
            self.stream = torch.cuda.Stream()   
        else:
            self._rref = None
        

    @property
    def data(self):
        return self._data
    
    @property
    def rref(self):
        return self._rref
    
    @property
    def ctx(self):
        return self._ctx
    @staticmethod
    @rpc.functions.async_execution
    def _index_selet(self):
        fut = torch.futures.Future()
        fut.set_result(None)
        return fut

    def all_gather_rrefs(self) -> List[rpc.RRef]:
        return self.ctx.all_gather_remote_objects(self.rref)

    def async_index_select(self, dim: int, index: Tensor, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_exec(TensorAccessor._index_select, rref, dim=dim, index=index)

    def async_index_copy_(self, dim: int, index: Tensor, source: Tensor, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_exec(TensorAccessor._index_copy_, rref, dim=dim, index=index, source=source)

    def async_index_add_(self, dim: int, index: Tensor, source: Tensor, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_exec(TensorAccessor._index_add_, rref, dim=dim, index=index, source=source)
    
    @staticmethod
    def _index_select(data: Tensor, dim: int, index: Tensor):
        stream = TensorAccessor.get_stream()
        with torch.cuda.stream(stream):
            data = data.index_select(dim, index)
            fut = torch.futures.Future()
            fut.set_result(data)
        return fut
    
    @staticmethod
    def _index_copy_(data: Tensor, dim: int, index: Tensor, source: Tensor):
        stream = TensorAccessor.get_stream()
        with torch.cuda.stream(stream):
            data.index_copy_(dim, index, source)
            fut = torch.futures.Future()
            fut.set_result(None)
        return fut

    @staticmethod
    def _index_add_(data: Tensor, dim: int, index: Tensor, source: Tensor):
        stream = TensorAccessor.get_stream()
        with torch.cuda.stream(stream):
            data.index_add_(dim, index, source)
            fut = torch.futures.Future()
            fut.set_result(None)
        return fut

    @staticmethod
    def get_stream() -> Optional[torch.cuda.Stream]:
        global _TENSOR_ACCESSOR_STREAM
        if torch.cuda.is_available():
            return None
        if _TENSOR_ACCESSOR_STREAM is None:
            _TENSOR_ACCESSOR_STREAM = torch.cuda.Stream()
        return _TENSOR_ACCESSOR_STREAM

_TENSOR_ACCESSOR_STREAM: Optional[torch.cuda.Stream] = None


class DistInt:
    def __init__(self, sizes: List[int]) -> None:
        self._data = tuple([int(t) for t in sizes])
        self._total = sum(self._data)
    
    def __getitem__(self, idx: int) -> int:
        return self._data[idx]
    
    def __call__(self) -> int:
        return self._total


class DistIndex:
    def __init__(self, index: Tensor, part_ids: Optional[Tensor] = None) -> None:
        if part_ids is None:
            self._data = index.long()
        else:
            index, part_ids = index.long(), part_ids.long()
            self._data = (index & 0xFFFFFFFFFFFF) | ((part_ids & 0xFFFF) << 50)
    
    @property
    def loc(self) -> Tensor:
        return self._data & 0xFFFFFFFFFFFF
    
    @property
    def part(self) -> Tensor:
        return (self._data >> 50) & 0xFFFF
    def set_shared(self):
        return (self._data | 0x1000000000000)
    def set_cached(self):
        return (self._data | 0x2000000000000)
    @property
    def is_cached(self):
        return ((self._data>>49)&1).to(torch.bool)
    @property
    def is_shared(self):
        return ((self._data>>48)&1).to(torch.bool)
    @property
    def dist(self) -> Tensor:
        return self._data
    
    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def device(self):
        return self._data.device
    
    def to(self,device) -> Tensor:
        return DistIndex(self._data.to(device))
    @property
    def shape(self):
        return self._data.shape
    def size(self,index):
        return self._data.size(index)

class DistributedTensor:
    def __init__(self, data: Tensor,cache_init_data:Tensor = None) -> None:
        self.accessor = TensorAccessor(data)
        if self.accessor.rref is not None: 
            self.rrefs = self.accessor.all_gather_rrefs()

            local_sizes = []
            for rref in self.rrefs:
                n = self.ctx.remote_call(Tensor.size, rref, dim=0).wait()
                local_sizes.append(n)
            self._num_nodes: int = sum(local_sizes)
            self._num_part_nodes: Tuple[int,...] = tuple(int(s) for s in local_sizes)
        else:
            self.rrefs = None
            #self._num_nodes: int = dist.get_world_size()
            #self._num_part_nodes:List = [torch.tensor(data.size(0),device = #data.device) for _ in range(self._num_nodes)]
            #dist.all_gather(self._num_part_nodes,torch.tensor(data.size(0),#device = data.device))
            #self._num_nodes = sum(self._num_part_nodes)
            
        self._part_id: int = self.accessor.ctx.rank
        self._num_parts: int = self.accessor.ctx.world_size
    
    @property
    def shape(self):
        return self.accessor.data.shape
    
    @property
    def dtype(self):
        return self.accessor.data.dtype
    
    @property
    def device(self):
        return self.accessor.data.device
    
    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    @property
    def num_part_nodes(self):# -> tuple[int,...]:
        return self._num_part_nodes
    
    @property
    def part_id(self) -> int:
        return self._part_id
    
    @property
    def num_parts(self) -> int:
        return self._num_parts
    
    def to(self,device):
        self.accessor._data = self.accessor.data.to(device)
        return self
    
    def __getitem__(self,index):
        return self.accessor.data[index]
    
    @property
    def ctx(self):
        return self.accessor.ctx
    
    @staticmethod
    def all_to_all_ind2ptr(dist_index: Union[Tensor, DistIndex],group = None) -> Dict[str, Union[List[int], Tensor]]:
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)
        send_ptr = torch.ops.torch_sparse.ind2ptr(dist_index.part, dist.get_world_size())
        send_sizes = send_ptr[1:] - send_ptr[:-1]
        recv_sizes = torch.empty_like(send_sizes)
        torch.distributed.all_to_all_single(recv_sizes, send_sizes,group=group)
        recv_ptr = torch.zeros(recv_sizes.numel() + 1).type_as(recv_sizes)
        recv_ptr[1:] = recv_sizes.cumsum(dim=0)
        send_ptr = send_ptr.tolist()
        recv_ptr = recv_ptr.tolist()

        recv_ind = torch.full((recv_ptr[-1],), (2**62-1)*2+1, dtype=dist_index.dtype, device=dist_index.device)
        all_to_all_s(recv_ind, dist_index.loc, recv_ptr, send_ptr,group=group)
        return {
            "send_ptr": send_ptr,
            "recv_ptr": recv_ptr,
            "recv_ind": recv_ind,
        }

    def all_to_all_ind2ptr(self, dist_index: Union[Tensor, DistIndex],send_index: Optional[Tensor] = None,group = None) -> Dict[str, Union[List[int], Tensor]]:
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)
        send_ptr = torch.ops.torch_sparse.ind2ptr(dist_index.part, self.num_parts)
        send_sizes = send_ptr[1:] - send_ptr[:-1]
        recv_sizes = torch.empty_like(send_sizes)
        torch.distributed.all_to_all_single(recv_sizes, send_sizes,group=group)
        recv_ptr = torch.zeros(recv_sizes.numel() + 1).type_as(recv_sizes)
        recv_ptr[1:] = recv_sizes.cumsum(dim=0)
        send_ptr = send_ptr.tolist()
        recv_ptr = recv_ptr.tolist()

        recv_ind = torch.full((recv_ptr[-1],), (2**62-1)*2+1, dtype=dist_index.dtype, device=dist_index.device)
        if send_index is None:
            all_to_all_s(recv_ind, dist_index.loc, recv_ptr, send_ptr,group=group)
            return {
                "send_ptr": send_ptr,
                "recv_ptr": recv_ptr,
                "recv_ind": recv_ind,
            }
        else:
            return {
                "send_ptr": send_ptr,
                "recv_ptr": recv_ptr,
                "send_ind": send_index,
            }


        
    def all_to_all_get(self,
        dist_index: Union[Tensor, DistIndex, None] = None,
        send_ptr: Optional[List[int]] = None,
        recv_ptr: Optional[List[int]] = None,
        recv_ind: Optional[List[int]] = None,
        is_async: bool = False,
        group = None,
        pin_func = None,
        device = torch.device('cuda')
    ) -> Tensor:
        if dist_index is not None:
            dist_dict = self.all_to_all_ind2ptr(dist_index)
            send_ptr = dist_dict["send_ptr"]
            recv_ptr = dist_dict["recv_ptr"]
            recv_ind = dist_dict["recv_ind"]
        if pin_func is not None:
            pin_mem = pin_func(0,recv_ind.shape[0])
            torch.index_select(self.accessor.data,0,recv_ind.to('cpu'),out=pin_mem)
            data = pin_mem.to(device,non_blocking = True)
        else:
            data = self.accessor.data[recv_ind].contiguous()
        recv = torch.empty(send_ptr[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
        if is_async:
            fut = all_to_all_s(recv, data, send_ptr, recv_ptr,async_op=is_async,group=group)
            fut2 = all_to_all_s(recv, data, send_ptr, recv_ptr,async_op=is_async,group=group)
            return recv,(fut,fut2)
        else:
            all_to_all_s(recv, data, send_ptr, recv_ptr,async_op=is_async,group=group)
            return recv
    @staticmethod
    def all_to_all_get_data(
        data:torch.Tensor,
        send_ptr: Optional[List[int]] = None,
        recv_ptr: Optional[List[int]] = None,
        is_async: bool = False,
        group = None,
    ) -> Tensor:
        recv = torch.empty(send_ptr[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
        if is_async:
            fut = all_to_all_s(recv, data, send_ptr, recv_ptr,async_op=is_async,group=group)
            return recv,fut
        else:
            all_to_all_s(recv, data, send_ptr, recv_ptr,async_op=is_async,group=group)
            return recv
        
    def all_to_all_set(self,
        data: Tensor,
        dist_index: Union[Tensor, DistIndex, None] = None,
        send_ptr: Optional[List[int]] = None,
        recv_ptr: Optional[List[int]] = None,
        recv_ind: Optional[List[int]] = None,
        group = None
    ):
        if dist_index is not None:
            dist_dict = self.all_to_all_ind2ptr(dist_index)
            send_ptr = dist_dict["send_ptr"]
            recv_ptr = dist_dict["recv_ptr"]
            recv_ind = dist_dict["recv_ind"]
        recv = torch.empty(recv_ptr[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
        all_to_all_s(recv, data.contiguous(), recv_ptr, send_ptr,group=group)
        self.accessor.data.index_copy_(0, recv_ind, recv)

    def all_to_all_send(self,
        dist_index: Union[Tensor, DistIndex, None] = None,
        send_ptr: Optional[List[int]] = None,
        recv_ptr: Optional[List[int]] = None,
        send_ind: Optional[List[int]] = None,
        group = None
    ):
        if dist_index is not None:
            dist_dict = self.all_to_all_ind2ptr(dist_index,send_index=send_ind)
            send_ptr = dist_dict["send_ptr"]
            recv_ptr = dist_dict["recv_ptr"]
        data = self.accessor.data.contiguous()
        data = data[send_ind].contiguous()
        recv = torch.empty(recv_ptr[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
        all_to_all_s(recv, data, recv_ptr, send_ptr,group=group)
        return recv
        
    def index_select(self, dist_index: Union[Tensor, DistIndex]):
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)

        part_idx = dist_index.part
        index = dist_index.loc

        futs: List[torch.futures.Future] = []
        for i in range(self.num_parts):
            f = self.accessor.async_index_select(0, index[part_idx == i], self.rrefs[i])
            futs.append(f)

        def callback(fs: torch.futures.Future[List[torch.futures.Future]]) -> Tensor:
            result: Optional[Tensor] = None
            for i, f in enumerate(fs.value()):
                t: Tensor = f.value()
                if result is None:
                    result = torch.empty(
                        part_idx.size(0), *t.shape[1:], dtype=t.dtype, device=t.device,
                    )
                
                result[part_idx == i] = t
            return result
        return torch.futures.collect_all(futs).then(callback)
    
    def index_copy_(self, dist_index: Union[Tensor, DistIndex], source: Tensor):
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)

        part_idx = dist_index.part
        index = dist_index.loc

        futs: List[torch.futures.Future] = []
        for i in range(self.num_parts):
            mask = part_idx == i
            f = self.accessor.async_index_copy_(0, index[mask], source[mask], self.rrefs[i])
            futs.append(f)
        return torch.futures.collect_all(futs)
    
    def index_add_(self, dist_index: Union[Tensor, DistIndex], source: Tensor):
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)

        part_idx = dist_index.part
        index = dist_index.loc

        futs: List[torch.futures.Future] = []
        for i in range(self.num_parts):
            mask = part_idx == i
            f = self.accessor.async_index_add_(0, index[mask], source[mask], self.rrefs[i])
            futs.append(f)
        return torch.futures.collect_all(futs)
    