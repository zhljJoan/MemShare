import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc

from torch import Tensor
from typing import *

from starrygl.parallel import all_gather_remote_objects
from .utils import init_local_edge_index


class Route(nn.Module):
    def __init__(self,
        src_ids: Tensor,
        dst_size: int,
    ) -> None:
        super().__init__()
        self.register_buffer("_src_ids", src_ids, persistent=False)
        self.dst_size = dst_size
        self._init_nids_mapper()
        self._init_part_mapper()
    
    @staticmethod
    def from_edge_index(dst_ids: Tensor, edge_index: Tensor):
        src_ids, local_edge_index = init_local_edge_index(dst_ids, edge_index)
        return Route(src_ids, dst_ids.size(0)), local_edge_index
    
    @property
    def src_ids(self) -> Tensor:
        return self.get_buffer("_src_ids")
    
    @property
    def src_size(self) -> int:
        return self.src_ids.size(0)
    
    @property
    def dst_ids(self) -> Tensor:
        return self.src_ids[:self.dst_size]
    
    @property
    def ext_ids(self) -> Tensor:
        return self.src_ids[self.dst_size:]
    
    @property
    def ext_size(self) -> int:
        return self.src_size - self.dst_size
    
    def parts_iter(self, local_ids: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
        world_size = dist.get_world_size()
        part_mapper = self.part_mapper[local_ids]
        for i in range(world_size):
            # part_ids = local_ids[part_mapper == i]
            part_ids = torch.where(part_mapper == i)[0]
            glob_ids = self.src_ids[part_ids]
            yield part_ids, glob_ids
    
    def to_local_ids(self, ids: Tensor) -> Tensor:
        return self.nids_mapper[ids]
    
    def _init_nids_mapper(self):
        num_nodes: int = self.src_ids.max().item() + 1
        device: torch.device = self.src_ids.device
        mapper = torch.empty(num_nodes, dtype=torch.long, device=device).fill_((2**62-1)*2+1)
        mapper[self.src_ids] = torch.arange(self.src_ids.size(0), dtype=torch.long, device=device)
        self.register_buffer("nids_mapper", mapper, persistent=False)
    
    def _init_part_mapper(self):
        device: torch.device = self.src_ids.device
        nids_mapper = self.get_buffer("nids_mapper")

        mapper = torch.empty(self.src_size, dtype=torch.int32, device=device).fill_(-1)
        for i, dst_ids in enumerate(all_gather_remote_objects(self.dst_ids)):
            dst_ids: Tensor = dst_ids.to_here().to(device)
            dst_ids = dst_ids[dst_ids < nids_mapper.size(0)]
            
            dst_local_inds = nids_mapper[dst_ids]
            dst_local_mask = dst_local_inds != ((2**62-1)*2+1)

            dst_local_inds = dst_local_inds[dst_local_mask]
            mapper[dst_local_inds] = i
        assert (mapper >= 0).all()
        self.register_buffer("part_mapper", mapper, persistent=False)

# class RouteTable(nn.Module):
#     def __init__(self,
#         src_ids: Tensor,
#         dst_size: int,
#     ) -> None:
#         super().__init__()
#         self.register_buffer("src_ids", src_ids)
#         self.src_size: int = src_ids.size(0)
#         self.dst_size = dst_size
#         assert self.src_size >= self.dst_size

#         self._init_mapper()

#         rank, world_size = rank_world_size()
#         rrefs = all_gather_remote_objects(self)

#         gather_futures: List[torch.futures.Future] = []
#         for i in range(world_size):
#             rref = rrefs[i]
#             fut = rpc.rpc_async(rref.owner(), RouteTable._get_dst_ids, args=(rref,))
#             gather_futures.append(fut)
        
#         max_src_ids: int = src_ids.max().item()
#         smp = torch.empty(max_src_ids + 1, dtype=torch.long, device=src_ids.device).fill_((2**62-1)*2+1)
#         smp[src_ids] = torch.arange(src_ids.size(0), dtype=smp.dtype, device=smp.device)

#         self.fw_masker = RouteMasker(self.dst_size, world_size)
#         self.bw_masker = RouteMasker(self.src_size, world_size)
#         dist.barrier()

#         scatter_futures: List[torch.futures.Future] = []
#         for i in range(world_size):
#             fut = gather_futures[i]
#             s_ids: Tensor = src_ids
#             d_ids: Tensor = fut.wait()

#             num_ids: int = max(s_ids.max().item(), d_ids.max().item()) + 1
#             imp = torch.zeros(num_ids, dtype=torch.long, device=self._get_device())
#             imp[s_ids] += 1
#             imp[d_ids] += 1
#             ind = torch.where(imp > 1)[0]
            
#             imp.fill_((2**62-1)*2+1)
#             imp[d_ids] = torch.arange(d_ids.size(0), dtype=imp.dtype, device=imp.device)

#             s_ind = smp[ind]
#             d_ind = imp[ind]

#             rref = rrefs[i]
#             fut = rpc.rpc_async(rref.owner(), RouteTable._set_fw_mask, args=(rref, rank, d_ind))
#             scatter_futures.append(fut)

#             bw_mask = torch.zeros(self.src_size, dtype=torch.bool).index_fill_(0, s_ind, 1)
#             self.bw_masker.set_mask(i, bw_mask)
        
#         for fut in scatter_futures:
#             fut.wait()
#         dist.barrier()

#     # def fw_index(self, dst: int, index: Tensor) -> Tensor:
#     #     mask = self.fw_masker.select(dst, index)
#     #     return self.get_global_index(index[mask])
    
#     # def bw_index(self, dst: int, index: Tensor) -> Tensor:
#     #     mask = self.bw_masker.select(dst, index)
#     #     return self.get_global_index(index[mask])

#     def fw_value_index(self, dst: int, value: Tensor, index: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
#         if index is None:
#             assert value.size(0) == self.dst_size
#             mask = self.fw_masker.select(dst)
#             return value[mask], self.get_buffer("src_ids")[:self.dst_size][mask]
#         else:
#             assert value.size(0) == index.size(0)
#             mask = self.fw_masker.select(dst, index)
#             value, index = value[mask], index[mask]
#             return value, self.get_global_index(index)
    
#     def bw_value_index(self, dst: int, value: Tensor, index: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
#         if index is None:
#             assert value.size(0) == self.src_size
#             mask = self.bw_masker.select(dst)
#             return value[mask], self.get_buffer("src_ids")[mask]
#         else:
#             assert value.size(0) == index.size(0)
#             mask = self.bw_masker.select(dst, index)
#             value, index = value[mask], index[mask]
#             return value, self.get_global_index(index)
    
#     def get_global_index(self, index: Tensor) -> Tensor:
#         return self.get_buffer("src_ids")[index]
    
#     def get_local_index(self, index: Tensor) -> Tensor:
#         return self.get_buffer("mapper")[index]

#     @staticmethod
#     def _get_dst_ids(rref: rpc.RRef):
#         self: RouteTable = rref.local_value()
#         src_ids = self.get_buffer("src_ids")
#         return src_ids[:self.dst_size]

#     @staticmethod
#     def _set_fw_mask(rref: rpc.RRef, dst: int, fw_ind: Tensor):
#         self: RouteTable = rref.local_value()
#         fw_mask = torch.zeros(self.dst_size, dtype=torch.bool).index_fill_(0, fw_ind, 1)
#         self.fw_masker.set_mask(dst, fw_mask)
    
#     def _get_device(self):
#         return self.get_buffer("src_ids").device
    
#     def _init_mapper(self):
#         src_ids = self.get_buffer("src_ids")
#         num_nodes: int = src_ids.max().item() + 1
#         mapper = torch.empty(num_nodes, dtype=torch.long, device=src_ids.device).fill_((2**62-1)*2+1)
#         mapper[src_ids] = torch.arange(src_ids.size(0), dtype=torch.long)
#         self.register_buffer("mapper", mapper)



# class RouteMasker(nn.Module):
#     def __init__(self,
#         num_nodes: int,
#         world_size: int,
#     ) -> None:
#         super().__init__()
#         m = (world_size + 7) // 8
#         self.num_nodes = num_nodes
#         self.world_size = world_size
#         self.register_buffer("data", torch.zeros(m, num_nodes, dtype=torch.uint8))
    
#     def forward(self, i: int, index: Optional[Tensor] = None) -> Tensor:
#         return self.select(i, index)
    
#     def select(self, i: int, index: Optional[Tensor] = None) -> Tensor:
#         i, data = self._idx_data(i)
#         k, r = i // 8, i % 8
#         if index is None:
#             mask = data[k].bitwise_right_shift(r).bitwise_and_(1)
#         else:
#             mask = data[k][index].bitwise_right_shift_(r).bitwise_and_(1)
#         return mask.type(dtype=torch.bool)

#     def set_mask(self, i: int, mask: Tensor) -> Tensor:
#         assert mask.size(0) == self.num_nodes
#         i, data = self._idx_data(i)
#         k, r = i // 8, i % 8
#         data[k] &= ~(1<<r)
#         data[k] |= mask.type(torch.uint8).bitwise_left_shift_(r)

#     def _idx_data(self, i: int) -> Tuple[int, Tensor]:
#         assert -self.world_size < i and i < self.world_size
#         i = (i + self.world_size) % self.world_size
#         return i, self.get_buffer("data")