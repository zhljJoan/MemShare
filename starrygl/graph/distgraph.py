import torch
import torch.nn as nn

from torch import Tensor
from typing import *

from contextlib import contextmanager
import torch_sparse

from .ndata import NData
from .edata import EData
from .utils import init_local_edge_index
from ..core import Route


class DistGraph:
    def __init__(self,
        ids: Tensor,
        edge_index: Tensor,
        args: Dict[str, Any] = {},
        device: Union[torch.device, str, int, None] = None,
    ):
        # graph's target device
        if device is None:
            device = ids.device
        self.device = torch.device(device)
        
        # build local_edge_index
        dst_ids = ids
        src_ids, local_edge_index = init_local_edge_index(
            dst_ids=dst_ids,
            edge_index=edge_index,
        )
        
        self._src_ids = src_ids
        self._dst_ids = dst_ids
        self._local_edge_index = local_edge_index
        self._local_edge_ptr: Optional[Tensor] = None

        # node's attributes
        self.ndata = NData(
            src_size=src_ids.size(0),
            dst_size=dst_ids.size(0),
        )
        
        # edge's attributes
        self.edata = EData(
            edge_size=local_edge_index.size(1),
        )

        # graph's attributes
        self.args = args

        # route table
        self.route = Route(dst_ids, src_ids)
    
    @property
    def edge_index(self) -> Tensor:
        if self._local_edge_index.device != self.device:
            self._local_edge_index = self._local_edge_index.to(self.device)
        return self._local_edge_index
    
    @property
    def edge_ptr(self) -> Optional[Tensor]:
        if self._local_edge_ptr is None:
            return None
        if self._local_edge_ptr.device != self.device:
            self._local_edge_ptr = self._local_edge_ptr.to(self.device)
        return self._local_edge_ptr
    
    @property
    def src_ids(self) -> Tensor:
        if self._src_ids.device != self.device:
            self._src_ids = self._src_ids.to(self.device)
        return self._src_ids
    
    @property
    def src_size(self) -> int:
        return self._src_ids.size(0)
    
    @property
    def dst_ids(self) -> Tensor:
        if self._dst_ids.device != self.device:
            self._dst_ids = self._dst_ids.to(self.device)
        return self._dst_ids
    
    @property
    def dst_size(self) -> int:
        return self._dst_ids.size(0)
    
    @torch.no_grad()
    def gather(self, values: Tensor, direct: str = "dst_to_src") -> Tensor:
        if direct == "dst_to_src":
            work = self.route.gather_forward(values, None, async_op=False, return_active=False)
        elif direct == "src_to_dst":
            work = self.route.gather_backward(values, None, async_op=False, return_active=False)
        else:
            raise ValueError(f"direct must be 'src_to_dst' or 'dst_to_src', but got '{direct}'")
        return work.get_values()
    
    @contextmanager
    def scoped_manager(self):
        stacked_ndata = self.ndata
        stacked_edata = self.edata
        try:
            self.ndata = NData(
                p=stacked_ndata,
            )
            self.edata = EData(
                p=stacked_edata,
            )
            yield self
        finally:
            self.ndata = stacked_ndata
            self.edata = stacked_edata
    
    # def permute_edge_(self):
    #     perm = self._local_edge_index[1].argsort()
    #     self._local_edge_index = self._local_edge_index[:,perm]
    #     self._local_edge_ptr = torch.ops.torch_sparse.ind2ptr(self._local_edge_index[1], self.dst_size)
    #     self.edata.permute_(perm)
    
    # @torch.no_grad()
    # def shrink(self,
    #     src_mask: Optional[Tensor] = None,
    #     dst_mask: Optional[Tensor] = None,
    # ):
    #     edge_index = self.edge_index
    #     device = edge_index.device
    #     ikw = dict(dtype=torch.long, device=device)
    #     bkw = dict(dtype=torch.bool, device=device)

    #     if src_mask is None and dst_mask is None:
    #         return self
    #     else:
    #         if dst_mask is None:
    #             dst_mask = torch.zeros(self.ndata.dst_size, **bkw)
    #         else:
    #             assert dst_mask.size(0) == self.ndata.dst_size
    #             dst_mask = dst_mask.clone()

    #         if src_mask is not None:
    #             m = src_mask[edge_index[0]]
    #             m = edge_index[1][m]
    #             dst_mask[m] = True

    #         edge_mask = dst_mask[edge_index[1]]
    #         edge_index = edge_index[:,edge_mask]

    #         src_mask = torch.zeros(self.ndata.src_size, **bkw)
    #         src_mask[:self.ndata.dst_size] = dst_mask
    #         src_mask[edge_index[0]] = True
    #         dst_mask = src_mask[:self.ndata.dst_size]

    #         # 重新编号edge_index
    #         imp = torch.empty(self.ndata.src_size, **ikw).fill_((2**62-1)*2+1)
    #         idx = torch.where(src_mask)[0]
    #         imp[idx] = torch.arange(idx.size(0), **ikw)
    #         edge_index = imp[edge_index.flatten()].view_as(edge_index)

    #         # assert dst_mask.count_nonzero() > edge_index[1].max().item()

    #         ndata = NData(
    #             src_size=None,
    #             dst_size=None,
    #             p=self.ndata,
    #             src_mask=src_mask,
    #             dst_mask=dst_mask,
    #         )

    #         edata = EData(
    #             edge_size=None,
    #             p=self.edata,
    #             edge_mask=edge_mask,
    #         )
    #         edata[EID] = edge_index

    #         return DistGraph(self.args, ndata, edata, self.route)


# class ShrinkGraph:
#     def __init__(self,
#         g: DistGraph,
#         src_mask: Optional[Tensor] = None,
#         dst_mask: Optional[Tensor] = None,
#     ) -> None:
#         device = g.edge_index.device
#         ikw = dict(dtype=torch.long, device=device)
#         bkw = dict(dtype=torch.bool, device=device)

#         if src_mask is None and dst_mask is None:
#             # 如果src_mask和dst_mask都不指定，
#             # 则ShrinkGraph是DistGraph的复刻
#             self.src_ids = g.sr
#             self.dst_size = g.dst_size
#             self.src_mask = torch.ones(self.src_size, **bkw)
#             self.dst_mask_size = g.dst_size
#             self.edge_index = g.edge_index
#             self.pgraph = g
#         else:
#             if src_mask is not None:
#                 tmp_mask = torch.zeros(g.dst_size, **bkw)
                
#                 # 计算直接激活的边
#                 m = src_mask[g.edge_index[0]]
#                 m = g.edge_index[1][m]

#                 # 计算直接激活的dst_ids
#                 tmp_mask[m] = True

#                 # 和已激活的dst_ids合并
#                 if dst_mask is not None:
#                     tmp_mask |= dst_mask
#                 dst_mask = tmp_mask
            
#             # 计算间接激活的边
#             edge_mask = dst_mask[g.edge_index[1]]
#             edge_index = g.edge_index[:,edge_mask]

#             # 计算间接激活的src_ids
#             src_mask = torch.zeros(g.src_size, **bkw)
#             src_mask[edge_index[0]] = True
#             src_mask[:g.dst_size] |= dst_mask

#             self.src_ids = g.src_ids[src_mask]
#             self.dst_size = src_mask[:g.dst_size].count_nonzero().item()
#             self.src_mask = src_mask
#             self.dst_mask_size = g.dst_size

#             # 重新编号edge_index
#             imp = torch.empty(g.src_size, **ikw).fill_((2**62-1)*2+1)
#             idx = torch.where(src_mask)[0]
#             imp[idx] = torch.arange(idx.size(0), **ikw)
#             edge_index = imp[edge_index.flatten()].view_as(edge_index)

#             self.edge_index = edge_index
#             self.pgraph = g
#             self.ndata = ShrinkData(self.dst_mask, g.ndata)
#             self.edata = ShrinkData(self.edge_mask, g.edata)