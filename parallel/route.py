import torch
import torch.autograd as autograd
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed.cclib import all_to_all_s, all_to_all_v


__all__ = [
    "Route",
    "RouteWork",
    "RouteWorkCache",
    "RouteAlltoAll",
]


class Route:
    @staticmethod
    def from_raw_indices(
        src_ids: Tensor,
        dst_ids: Tensor,
        bipartite: bool = True,
        group: Any = None,
    ) -> 'Route':
        if group is None:
            group = dist.GroupMember.WORLD
            
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
    
    def filter(self,
        dst_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        remap: bool = False,
    ):
        if dst_mask is None:
            if src_mask is None:
                raise ValueError("please provide at least one parameter.")
            else:
                assert src_mask.dtype == torch.bool
                assert src_mask.numel() == self.src_len
                dst_mask = self.bw_tensor(src_mask.long()) != 0
        else:
            assert dst_mask.dtype == torch.bool
            assert dst_mask.numel() == self.dst_len
            tmp_src_mask = self.fw_tensor(dst_mask.long()) != 0
            if src_mask is None:
                src_mask = tmp_src_mask
            else:
                tmp_src_mask &= src_mask
                src_mask = tmp_src_mask
                dst_mask = self.bw_tensor(src_mask.long()) != 0
        fw_ptr, fw_ind = Route.__filter_ind_and_ptr(self._fw_ptr, self._fw_ind, dst_mask)
        bw_ptr, bw_ind = Route.__filter_ind_and_ptr(self._bw_ptr, self._bw_ind, src_mask)
        route = Route(
            src_len=self.src_len,
            dst_len=self.dst_len,
            fw_ptr=fw_ptr, fw_ind=fw_ind,
            bw_ptr=bw_ptr, bw_ind=bw_ind,
            group=self.group,
        )

        if remap:
            fw_ind, dst_len = Route.__remap_ind(route._fw_ind, dst_mask)
            bw_ind, src_len = Route.__remap_ind(route._bw_ind, src_mask)
            route = Route(
                src_len=src_len,
                dst_len=dst_len,
                fw_ptr=route._fw_ptr, fw_ind=fw_ind,
                bw_ptr=route._bw_ptr, bw_ind=bw_ind,
                group=route.group,
            )
        return dst_mask, src_mask, route
    
    def rev(self):
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
        assert len(fw_ptr) == len(bw_ptr)
        self._src_len = src_len
        self._dst_len = dst_len
        self._fw_ptr = tuple(fw_ptr)
        self._fw_ind = fw_ind
        self._bw_ptr = tuple(bw_ptr)
        self._bw_ind = bw_ind
        self._group = group

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
    
    # def fw_table(self, i: int):
    #     return self._fw_ind[self._fw_ptr[i]:self._fw_ptr[i+1]]
    
    # def bw_table(self, i: int):
    #     return self._bw_ind[self._bw_ptr[i]:self._bw_ptr[i+1]]
    
    def apply(self,
        data: Tensor,
        cache: Optional['RouteWorkCache'] = None,
        cache_key: Optional[str] = None,
    ) -> Tensor:
        return RouteAlltoAll.apply(data, self, cache, cache_key)
    
    @torch.no_grad()
    def fw_tensor(self, data: Tensor, async_op: bool = False):
        assert data.size(0) == self.dst_len

        output_tensor = torch.empty(
            self._bw_ind.numel(), *data.shape[1:],
            dtype=data.dtype, device=data.device,
        )

        work = all_to_all_s(
            output_tensor, data[self._fw_ind],
            self._bw_ptr, self._fw_ptr,
            group=self.group,
            async_op=async_op,
        )
        work = RouteWork(
            work if async_op else None,
            self._bw_ptr, self._bw_ind,
            self.src_len, output_tensor,
        )
        return work if async_op else work.wait()
    
    @torch.no_grad()
    def bw_tensor(self, data: Tensor, async_op: bool = False):
        assert data.size(0) == self.src_len

        output_tensor = torch.empty(
            self._fw_ind.numel(), *data.shape[1:],
            dtype=data.dtype, device=data.device,
        )

        work = all_to_all_s(
            output_tensor, data[self._bw_ind],
            self._fw_ptr, self._bw_ptr,
            group=self.group,
            async_op=async_op,
        )
        work = RouteWork(
            work if async_op else None,
            self._fw_ptr, self._fw_ind,
            self.dst_len, output_tensor,
        )
        return work if async_op else work.wait()
    
    @torch.no_grad()
    def get_src_part_ids(self) -> Tensor:
        input_tensor = torch.full_like(self._fw_ind, self.part_id)
        output_tensor = torch.empty_like(self._bw_ind)

        all_to_all_s(
            output_tensor, input_tensor,
            self._bw_ptr, self._fw_ptr,
            group=self.group,
        )

        out = torch.full(
            (self.src_len,), 2**16-1,
            dtype=self._bw_ind.dtype,
            device=self._bw_ind.device,
        )
        for s, t in zip(self._bw_ptr, self._bw_ptr[1:]):
            ind = self._bw_ind[s:t]
            assert (out[ind] == 2**16-1).all(), f"some vertices exist in more than one partition"
            out[ind] = output_tensor[s:t] & 0xFF
        return out

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

        # all_reduce number of nodes
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
            
            src_rank = dist.get_global_rank(group, i)
            all_dst_get[i] = dist.broadcast(
                all_dst_ids[i], src=src_rank,
                async_op=True, group=group,
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

            fw_tables.append(dst_ind)
            bw_tables.append(src_ind)

        fw_tables = Route.__backward_fw_tables(fw_tables, group=group)
        
        # add self-loops if not bipartite graph
        if not bipartite:
            rank_ind = torch.arange(dst_len, **ikw)
            fw_tables[rank] = bw_tables[rank] = rank_ind
        return fw_tables, bw_tables
    
    @staticmethod
    def __filter_ind_and_ptr(ptr: List[int], ind: Tensor, mask: Tensor) -> Tuple[List[int], Tensor]:
        m = mask[ind]
        new_ptr: List[int] = [0]
        new_ind: List[Tensor] = []
        for s, t in zip(ptr, ptr[1:]):
            new_ind.append(ind[s:t][m[s:t]])
            new_ptr.append(new_ptr[-1] + new_ind[-1].numel())
        return new_ptr, torch.cat(new_ind, dim=0)
    
    @staticmethod
    def __remap_ind(ind: Tensor, mask: Tensor) -> Tuple[Tensor, int]:
        idx = torch.where(mask)[0]
        imp = torch.full((mask.numel(),), (2**62-1)*2+1, dtype=ind.dtype, device=ind.device)
        imp[idx] = torch.arange(idx.numel(), dtype=ind.dtype, device=ind.device)
        return imp[ind], idx.numel()

        # n: int = mask.count_nonzero().item()
        # imp = torch.full((mask.numel(),), (2**62-1)*2+1, dtype=ind.dtype, device=ind.device)
        # imp[mask] = torch.arange(n, dtype=ind.dtype, device=ind.device)
        # return ind, int(n)
    
    @staticmethod
    def __backward_fw_tables(
        fw_tables: List[Tensor],
        group: Any,
    ) -> List[Tensor]:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        send_sizes = [t.size() for t in fw_tables]
        recv_sizes = [None] * world_size

        dist.all_gather_object(recv_sizes, send_sizes, group=group)
        recv_sizes = [s[rank] for s in recv_sizes]

        fixed_tables = []
        for s, t in zip(recv_sizes, fw_tables):
            t = torch.empty(*s, dtype=t.dtype, device=t.device)
            fixed_tables.append(t)
        all_to_all_v(fixed_tables, fw_tables, group=group)
        return fixed_tables
    
    @staticmethod
    def __tables_to_indptr(
        fw_tables: List[Tensor],
        bw_tables: List[Tensor],
    ):
        fw_ptr: List[int] = [0]
        for t in fw_tables:
            assert t.dim() == 1
            fw_ptr.append(fw_ptr[-1] + t.numel())
        fw_ind = torch.cat(fw_tables, dim=0)
        
        bw_ptr: List[int] = [0]
        for t in bw_tables:
            assert t.dim() == 1
            bw_ptr.append(bw_ptr[-1] + t.numel())
        bw_ind = torch.cat(bw_tables, dim=0)

        return {
            "fw_ptr": fw_ptr, "bw_ptr": bw_ptr,
            "fw_ind": fw_ind, "bw_ind": bw_ind,
        }

class RouteWork:
    def __init__(self,
        work: Any,
        ptr: List[int],
        ind: Tensor,
        len: int,
        recv_t: Tensor,
    ) -> None:
        self._work = work
        self._ptr = ptr
        self._ind = ind
        self._len = len
        self._recv_t = recv_t

        if self._work is None:
            self._reduce()
    
    def _reduce(self):
        out = torch.zeros(
            self._len, *self._recv_t.shape[1:],
            dtype=self._recv_t.dtype,
            device=self._recv_t.device,
        )
        for s, t in zip(self._ptr, self._ptr[1:]):
            ind = self._ind[s:t]
            out[ind] += self._recv_t[s:t]

        self._work = None
        self._ptr = None
        self._ind = None
        self._len = None
        self._recv_t = out

    def wait(self) -> Tensor:
        if self._work is None:
            return self._recv_t
        self._work.wait()
        self._reduce()
        return self._recv_t


class RouteWorkCache:
    def __init__(self,
        enable_fw: bool = True,
        enable_bw: bool = True,
    ) -> None:
        self.enable_fw = enable_fw
        self.enable_bw = enable_bw
        self._cached_works: Dict[str, RouteWork] = {}
    
    def enable_fw_(self, enable: bool = True):
        self.enable_fw = enable
        return self
    
    def enable_bw_(self, enable: bool = True):
        self.enable_bw = enable
        return self
    
    def wait(self):
        for work in self._cached_works.values():
            work.wait()
    
    def clear(self):
        self._cached_works.clear()

    def get_and_set(self,
        key: str,
        work: RouteWork,
        bw: bool = False,
    ) -> Optional[RouteWork]:
        if bw and self.enable_bw:
            key = key + "_bw"
        elif not bw and self.enable_fw:
            key = key + "_fw"
        else:
            return work
        t = self._cached_works.get(key, work)
        self._cached_works[key] = work
        return t
    

class RouteAlltoAll(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        x: Tensor,
        route: Route,
        cache: Optional[RouteWorkCache],
        cache_key: Optional[str],
    ):
        ctx.saved_route = route
        ctx.saved_cache = cache
        ctx.saved_cache_key = cache_key
        if cache is None or cache_key is None:
            return route.fw_tensor(x)
        else:
            work = route.fw_tensor(x, async_op=True)
            work = cache.get_and_set(cache_key, work, bw=False)
            return work.wait()

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ) -> Tensor:
        route: Route = ctx.saved_route
        cache: Optional[RouteWorkCache] = ctx.saved_cache
        cache_key: Optional[str] = ctx.saved_cache_key

        if cache is None or cache_key is None:
            return route.bw_tensor(grad), None, None, None
        else:
            work = route.bw_tensor(grad, async_op=True)
            work = cache.get_and_set(cache_key, work, bw=True)
            return work.wait(), None, None, None
        