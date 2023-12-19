from typing import Any
import torch
import torch.distributed.rpc as rpc

from torch import Tensor
from torch.types import Number
from typing import *



class TensorAccessor:
    def __init__(self, data: Tensor) -> None:
        from .context import DistributedContext

        self._data = data
        self._ctx = DistributedContext.get_default_context()
        self._rref = rpc.RRef(data)
        self._rref.confirmed_by_owner

    @property
    def data(self):
        return self._data
    
    @property
    def rref(self):
        return self._rref
    
    @property
    def ctx(self):
        return self._ctx
    
    def all_gather_index(self,index,input_split) -> Tensor:
        out_split = torch.empty_like(input_split)
        torch.distributed.all_to_all_single(out_split,input_split)
        input_split = list(input_split)
        output = torch.empty([out_split.sum()],dtype = index.dtype,device = index.device)
        out_split = list(out_split)
        torch.distributed.all_to_all_single(output,index,out_split,input_split)
        return output,out_split,input_split
    
    def all_gather_data(self,index,input_split,out_split):
        output = torch.empty([int(Tensor(out_split).sum().item()),*self._data.shape[1:]],dtype = self._data.dtype,device = 'cuda')#self._data.device)
        torch.distributed.all_to_all_single(output,self.data[index.to(self.data.device)].to('cuda'),output_split_sizes = out_split,input_split_sizes = input_split)
        return output

    def all_gather_rrefs(self) -> List[rpc.RRef]:
        return self.ctx.all_gather_remote_objects(self.rref)

    def async_index_select(self, dim: int, index: Tensor, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_call(Tensor.index_select, rref, dim=dim, index=index)

    def async_index_copy_(self, dim: int, index: Tensor, source: Tensor, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_void_call(Tensor.index_copy_, rref, dim=dim, index=index, source=source)

    def async_index_add_(self, dim: int, index: Tensor, source: Tensor, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_void_call(Tensor.index_add_, rref, dim=dim, index=index, source=source)
    
    def async_index_fill_(self, dim: int, index: Tensor, value: Number, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_void_call(Tensor.index_fill_, rref, dim=dim, index=index, value=value)

    def async_fill_(self, value: Number, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        return self.ctx.remote_void_call(Tensor.fill_, rref, value=value)

    def async_zero_(self, rref: Optional[rpc.RRef] = None):
        if rref is None:
            rref = self.rref
        self.ctx.remote_void_call(Tensor.zero_, rref)


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
            self.data = index.long()
        else:
            index, part_ids = index.long(), part_ids.long()
            self.data = (index & 0xFFFFFFFFFFFF) | ((part_ids & 0xFFFF) << 48)
    
    @property
    def loc(self) -> Tensor:
        return self.data & 0xFFFFFFFFFFFF
    
    @property
    def part(self) -> Tensor:
        return (self.data >> 48).int() & 0xFFFF
    
    @property
    def dist(self) -> Tensor:
        return self.data
    def to(self,device) -> Tensor:
        return DistIndex(self.data.to(device))


class DistributedTensor:
    def __init__(self, data: Tensor) -> None:
        self.accessor = TensorAccessor(data)
        self.rrefs = self.accessor.all_gather_rrefs()
        # self.num_parts = len(self.rrefs)

        local_sizes = []
        for rref in self.rrefs:
            n = self.ctx.remote_call(Tensor.size, rref, dim=0).wait()
            local_sizes.append(n)
        self.num_nodes = DistInt(local_sizes)
        self.num_parts = DistInt([1] * len(self.rrefs))
        self.distptr = torch.tensor([((part_ids & 0xFFFF) << 48) for part_ids in range(self.num_parts()+1)],device = 'cuda')#data.device)


    @property
    def dtype(self):
        return self.accessor.data.dtype
    
    @property
    def device(self):
        return self.accessor.data.device
    
    def to(self,device):
        return self.accessor.data.to(device)
    
    def __getitem__(self,index):
        return self.accessor.data[index]
    
    @property
    def ctx(self):
        return self.accessor.ctx
    
    def gather_select_index(self,dist_index: Union[Tensor,DistIndex]):
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)
        data = dist_index.data
        posptr = torch.searchsorted(data,self.distptr,right = False)
        input_split = posptr[1:] - posptr[:-1]
        return self.accessor.all_gather_index(DistIndex(data).loc,input_split)
    
    def scatter_data(self,local_index,input_split,out_split):
        return self.accessor.all_gather_data(local_index,input_split=input_split,out_split=out_split)
    
    def index_select(self, dist_index: Union[Tensor, DistIndex]):
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)

        part_idx = dist_index.part
        index = dist_index.loc

        futs: List[torch.futures.Future] = []
        for i in range(self.num_parts()):
            #if i != torch.distributed.get_rank():
            #    continue
            #f = torch.futures.Future()
            #f.set_result(self.accessor.data[index[part_idx == i]])
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
        for i in range(self.num_parts()):
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
        for i in range(self.num_parts()):
            mask = part_idx == i
            f = self.accessor.async_index_add_(0, index[mask], source[mask], self.rrefs[i])
            futs.append(f)
        return torch.futures.collect_all(futs)
    
    def index_fill_(self, dist_index: Union[Tensor, DistIndex], value: Number):
        if isinstance(dist_index, Tensor):
            dist_index = DistIndex(dist_index)

        part_idx = dist_index.part
        index = dist_index.loc

        futs: List[torch.futures.Future] = []
        for i in range(self.num_parts()):
            mask = part_idx == i
            f = self.accessor.async_index_fill_(0, index[mask], value, self.rrefs[i])
            futs.append(f)
        return torch.futures.collect_all(futs)
