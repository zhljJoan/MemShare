import torch
import torch.autograd as autograd
import torch.distributed as dist

from multiprocessing.pool import ThreadPool

from torch import Tensor
from typing import *

from starrygl.distributed import DistributedContext



class Route:
    @staticmethod
    def from_raw_indices(
        src_ids: Tensor,
        dst_ids: Tensor,
        bipartite: bool = True,
        group: Any = None,
    ) -> 'Route':
        fw_tables, bw_tables = Route._build_route_tables(
            src_ids=src_ids, dst_ids=dst_ids,
            bipartite=bipartite, group=group,
        )
        
        return Route(
            src_len=src_ids.size(0),
            dst_len=dst_ids.size(0),
            **Route.__tables_to_indptr(fw_tables, bw_tables),
            group=group,
        )
    
    def new_subroute(self, dst_mask: Tensor):
        if dst_mask.dtype != torch.bool:
            dst_mask = Route.__expand_index_to_mask(dst_mask, self.dst_len)
        assert dst_mask.size(0) == self.dst_len

        fw_dst_masks, work = self.all_to_all_fw(
            input_tensor_list=[dst_mask[self.fw_table(i)[0]] for i in range(self.num_parts)],
            async_op=True,
        )
        
        fw_tables: List[Tensor] = []
        for i in range(self.num_parts):
            m = dst_mask[self.fw_table(i)[0]]
            fw_tables.append(self.fw_table(i)[:,m])
        
        work.wait()

        bw_tables: List[Tensor] = []
        for i, m in enumerate(fw_dst_masks):
            bw_tables.append(self.bw_table(i)[:,m])
        
        return Route(
            src_len=self.src_len,
            dst_len=self.dst_len,
            **Route.__tables_to_indptr(fw_tables, bw_tables),
            group=self.group,
        )
    
    def reverse_route(self):
        return Route(
            src_len=self.dst_len,
            dst_len=self.src_len,
            fw_ptr=self._bw_ptr, fw_ind=self._bw_ind,
            bw_ptr=self._fw_ptr, bw_ind=self._fw_ind,
            group=self.group,
        )
    
    def __init__(self,
        src_len: int, dst_len: int,
        fw_ptr: List[int], fw_ind: Tensor,
        bw_ptr: List[int], bw_ind: Tensor,
        group: Any,
    ) -> None:
        self._ctx = DistributedContext.get_default_context()
        self._src_len = src_len
        self._dst_len = dst_len
        self._fw_ptr = fw_ptr
        self._fw_ind = fw_ind
        self._bw_ptr = bw_ptr
        self._bw_ind = bw_ind
        self._group = group
    
    @property
    def ctx(self):
        return self._ctx
    
    @property
    def group(self):
        return self._group
    
    @property
    def src_len(self):
        return self._src_len
    
    @property
    def dst_len(self):
        return self._dst_len
    
    @property
    def part_id(self) -> int:
        return dist.get_rank(self.group)
    
    @property
    def num_parts(self) -> int:
        return dist.get_world_size(self.group)
    
    def to(self, device: Any):
        self._fw_ind = self._fw_ind.to(device)
        self._bw_ind = self._bw_ind.to(device)
        return self
    
    def fw_table(self, i: int):
        return self._fw_ind[:,self._fw_ptr[i]:self._fw_ptr[i+1]]
    
    def bw_table(self, i: int):
        return self._bw_ind[:,self._bw_ptr[i]:self._bw_ptr[i+1]]
    
    def apply(self, data: Tensor) -> Tensor:
        return RouteAlltoAll.apply(data, self)
    
    def fw_tensor(self, data: Tensor) -> Tensor:
        assert data.size(0) == self.dst_len
        xs = self.all_to_all_fw(
            [data[self.fw_table(i)[0]] for i in range(self.num_parts)],
            async_op=False,
        )

        out = torch.zeros(
            self.src_len, *data.shape[1:],
            dtype=data.dtype, device=data.device,
        )
        for i, t in enumerate(xs):
            out[self.bw_table(i)[0]] += t
        return out
    
    def bw_tensor(self, data: Tensor) -> Tensor:
        assert data.size(0) == self.src_len
        xs = self.all_to_all_bw(
            [data[self.bw_table(i)[0]] for i in range(self.num_parts)],
            async_op=False,
        )

        out = torch.zeros(
            self.dst_len, *data.shape[1:],
            dtype=data.dtype, device=data.device,
        )
        for i, t in enumerate(xs):
            out[self.fw_table(i)[0]] += t
        return out
    
    def all_to_all_fw(self, input_tensor_list: List[Tensor], async_op: bool = False):
        output_tensor_list: List[Tensor] = []
        for i in range(self.num_parts):
            t = input_tensor_list[i]
            assert t.size(0) == self._fw_ptr[i+1] - self._fw_ptr[i]

            s = self._bw_ptr[i+1] - self._bw_ptr[i]
            output_tensor_list.append(
                torch.empty(s, *t.shape[1:], dtype=t.dtype, device=t.device)
            )
        
        work = self.ctx.all_to_all_v(
            output_tensor_list,
            input_tensor_list,
            group=self.group, async_op=async_op,
        )

        if async_op:
            return output_tensor_list, work
        else:
            return output_tensor_list
    
    def all_to_all_bw(self, input_tensor_list: List[Tensor], async_op: bool = False):
        output_tensor_list: List[Tensor] = []
        for i in range(self.num_parts):
            t = input_tensor_list[i]
            assert t.size(0) == self._bw_ptr[i+1] - self._bw_ptr[i]
            
            s = self._fw_ptr[i+1] - self._fw_ptr[i]
            output_tensor_list.append(
                torch.empty(s, *t.shape[1:], dtype=t.dtype, device=t.device)
            )
        
        work = self.ctx.all_to_all_v(
            output_tensor_list,
            input_tensor_list,
            group=self.group, async_op=async_op,
        )

        if async_op:
            return output_tensor_list, work
        else:
            return output_tensor_list

    @staticmethod
    def _build_route_tables(
        src_ids: Tensor,
        dst_ids: Tensor,
        bipartite: bool,
        group: Any,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        assert src_ids.dtype == torch.long
        assert dst_ids.dtype == torch.long
        assert src_ids.dim() == 1
        assert dst_ids.dim() == 1

        ctx = DistributedContext.get_default_context()

        src_len = src_ids.size(0)
        dst_len = dst_ids.size(0)
        
        if not bipartite:
            assert dst_len <= src_len
            assert (src_ids[:dst_len] == dst_ids).all()
        
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        ikw = dict(dtype=torch.long, device=dst_ids.device)

        all_dst_lens: List[int] = [None] * world_size
        dist.all_gather_object(all_dst_lens, dst_len, group=group)

        num_nodes = torch.zeros(1, **ikw)
        if src_ids.numel() > 0:
            num_nodes = src_ids.max().max(num_nodes)
        if dst_ids.numel() > 0:
            num_nodes = dst_ids.max().max(num_nodes)
        dist.all_reduce(num_nodes, op=dist.ReduceOp.MAX, group=group)
        num_nodes = num_nodes.item() + 1

        # src_ids -> local index
        smp: Tensor = torch.empty(num_nodes, **ikw).fill_((2**62-1)*2+1)
        smp[src_ids] = torch.arange(src_len, **ikw)
        
        # async fetch dst_ids from other partitions
        all_dst_ids: List[Tensor] = [None] * world_size
        all_dst_get = [None] * world_size
        def async_fetch(i: int):
            if i == rank:
                all_dst_ids[i] = dst_ids
            else:
                all_dst_ids[i] = torch.empty(all_dst_lens[i], **ikw)
            all_dst_get[i] = dist.broadcast(
                all_dst_ids[i], src=i, async_op=True, group=group
            )
        
        fw_tables: List[Tensor] = []
        bw_tables: List[Tensor] = []

        xmp = torch.empty_like(smp)
        for i in range(world_size):
            # prefetch dst_ids
            if i == 0:
                async_fetch(i)
            if i + 1 < world_size:
                async_fetch(i + 1)
            all_dst_get[i].wait()
            ids = all_dst_ids[i]

            xmp.fill_(0)
            xmp[ids] += 1
            xmp[src_ids] += 1
            ind = torch.where(xmp > 1)[0]

            # dst_ids -> local index
            xmp.fill_((2**62-1)*2+1)
            xmp[ids] = torch.arange(ids.size(0), **ikw)
            
            # remap src_ids and dst_ids
            src_ind = smp[ind]
            dst_ind = xmp[ind]

            # 此时只有bw_route是正常的，fw_route需要发送给src_ids所在分区
            fw_tables.append(torch.vstack([dst_ind, src_ind]))
            bw_tables.append(torch.vstack([src_ind, dst_ind]))
        
        fw_tables = [t.t().contiguous() for t in fw_tables]
        fw_tables = ctx.all_to_all_g(fw_tables, group=group)
        fw_tables = [t.t().contiguous() for t in fw_tables]
        
        # 非二分图，每个点添加自环
        if not bipartite:
            rank_ind = torch.arange(dst_len, **ikw)
            fw_tables[rank] = bw_tables[rank] = torch.vstack([rank_ind, rank_ind])
        return fw_tables, bw_tables
    
    @staticmethod
    def __expand_index_to_mask(index: Tensor, len: int) -> Tensor:
        mask = torch.zeros(len, dtype=torch.bool, device=index.device)
        mask[index] = True
        return mask
    
    @staticmethod
    def __tables_to_indptr(
        fw_tables: List[Tensor],
        bw_tables: List[Tensor],
    ):
        fw_ptr: List[int] = [0]
        for t in fw_tables:
            last_n = fw_ptr[-1]
            fw_ptr.append(last_n + t.size(1))
        fw_ind = torch.cat(fw_tables, dim=1)
        
        bw_ptr: List[int] = [0]
        for t in bw_tables:
            last_n = bw_ptr[-1]
            bw_ptr.append(last_n + t.size(1))
        bw_ind = torch.cat(bw_tables, dim=1)

        return {
            "fw_ptr": fw_ptr, "bw_ptr": bw_ptr,
            "fw_ind": fw_ind, "bw_ind": bw_ind,
        }


class RouteAlltoAll(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        x: Tensor,
        route: Route,
    ):
        ctx.saved_route = route
        return route.fw_tensor(x)
        

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ) -> Tensor:
        route: Route = ctx.saved_route
        return route.bw_tensor(grad)