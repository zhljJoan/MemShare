from typing import Any
import torch
import torch.distributed as dist
import torch.autograd as autograd

from torch import Tensor
from typing import *

from torch_sparse import SparseTensor


__all__ = [
    "SparseBlocks",
]

class SparseBlocks:
    @staticmethod
    def from_raw_indices(
        dst_ids: Tensor,
        edge_index: Tensor,
        src_ids: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        group: Any = None,
    ) -> 'SparseBlocks':
        assert edge_index.dim() == 2 and edge_index.size(0) == 2

        if src_ids is None:
            src_ids = dst_ids
        
        src_ids, src_ptr = SparseBlocks.__fetch_ids_sizes(src_ids, group=group)
        adj_ts = SparseBlocks.__remap_adj_t(dst_ids, edge_index, src_ids, src_ptr, edge_attr)

        return SparseBlocks(adj_ts, group=group)
    
    def __init__(self, adj_ts: List[SparseTensor], group: Any) -> None:
        self._adj_ts = adj_ts
        self._group = group
    
    def adj_t(self, i: int) -> SparseTensor:
        return self._adj_ts[i]
    
    @property
    def group(self):
        return self._group
    
    @property
    def part_id(self) -> int:
        return dist.get_rank(self._group)
    
    @property
    def num_parts(self) -> int:
        return dist.get_world_size(self._group)
    
    def apply(self, x: Tensor) -> Tensor:
        return SparseBlockMM.apply(self, x)

    @staticmethod
    def __fetch_ids_sizes(local_ids: Tensor, group: Any):
        assert local_ids.dim() == 1

        if group is None:
            group = dist.GroupMember.WORLD

        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        ikw = dict(dtype=torch.long, device=local_ids.device)

        all_lens: List[int] = [None] * world_size
        dist.all_gather_object(all_lens, local_ids.numel(), group=group)

        # all reduce num_nodes
        num_nodes = torch.zeros(1, **ikw)
        if local_ids.numel() > 0:
            num_nodes = local_ids.max().max(num_nodes)
        dist.all_reduce(num_nodes, op=dist.ReduceOp.MAX, group=group)
        num_nodes: int = num_nodes.item() + 1

        # async fetch remote ids
        all_ids: List[Tensor] = [None] * world_size
        all_get = [None] * world_size
        def async_fetch(i: int):
            if i == rank:
                all_ids[i] = local_ids
            else:
                all_ids[i] = torch.empty(all_lens[i], **ikw)
            src = dist.get_global_rank(group, i)
            all_get[i] = dist.broadcast(
                all_ids[i], src=src, async_op=True, group=group
            )
        
        imp: Tensor = torch.full((num_nodes,), (2**62-1)*2+1, **ikw)

        offset: int = 0
        for i in range(world_size):
            if i == 0:
                async_fetch(i)
            if i + 1 < world_size:
                async_fetch(i + 1)
            all_get[i].wait()
            ids = all_ids[i]

            assert (imp[ids] == (2**62-1)*2+1).all(), "all ids must be orthogonal."
            imp[ids] = torch.arange(offset, offset + all_lens[i], **ikw)
            offset += all_lens[i]
        assert (imp != (2**62-1)*2+1).all(), "some points that do not exist."
        ids = torch.cat(all_ids, dim=0)

        ptr: List[int] = [0]
        for s in all_lens:
            ptr.append(ptr[-1] + s)

        return ids, ptr
    
    @staticmethod
    def __remap_adj_t(
        dst_ids: Tensor,
        edge_index: Tensor,
        src_ids: Tensor,
        src_ptr: List[int],
        edge_attr: Optional[Tensor],
    ) -> List[SparseTensor]:
        ikw = dict(dtype=torch.long, device=dst_ids.device)
        imp: Tensor = torch.full((dst_ids.max().item()+1,), (2**62-1)*2+1, **ikw)
        imp[dst_ids] = torch.arange(dst_ids.numel(), **ikw)
        
        dst = imp[edge_index[1]]
        assert (dst != (2**62-1)*2+1).all()

        imp: Tensor = torch.full((src_ids.max().item()+1,), (2**62-1)*2+1, **ikw)
        imp[src_ids] = torch.arange(src_ids.numel(), **ikw)

        src = imp[edge_index[0]]
        assert (src != (2**62-1)*2+1).all()

        edge_index = torch.vstack([src, dst])
        adj = SparseTensor.from_edge_index(
            edge_index=edge_index,
            edge_attr=edge_attr,
            sparse_sizes=(src_ids.numel(), dst_ids.numel()),
        )

        adj_ts: List[SparseTensor] = []
        for s, t in zip(src_ptr, src_ptr[1:]):
            adj_ts.append(adj[s:t].t())
        return adj_ts


class SparseBlockMM(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        sp: SparseBlocks,
        x: Tensor,
    ):
        part_id = sp.part_id
        num_parts = sp.num_parts

        group = sp.group
        if group is None:
            group = dist.GroupMember.WORLD

        def async_fetch(i: int):
            n = sp.adj_t(i).sparse_size(1)
            if i == part_id:
                h = x.clone()
            else:
                h = torch.empty(n, *x.shape[1:], dtype=x.dtype, device=x.device)
            src = dist.get_global_rank(group, i)
            return dist.broadcast(h, src=src, group=sp.group, async_op=True)

        last_work = None
        out = None
        for i in range(num_parts):
            if i == 0:
                work = async_fetch(0)
            else:
                work = last_work

            if i + 1 < sp.num_parts:
                last_work = async_fetch(i + 1)
            
            work.wait()
            h, = work.result()

            if out is None:
                out = sp.adj_t(i) @ h
            else:
                out += sp.adj_t(i) @ h
        
        ctx.saved_sp = sp
        return out


    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
    ):
        sp: SparseBlocks = ctx.saved_sp

        part_id = sp.part_id
        num_parts = sp.num_parts

        group = sp.group
        if group is None:
            group = dist.GroupMember.WORLD

        def async_reduce(i: int, g: Tensor):
            dst = dist.get_global_rank(group, i)
            return dist.reduce(
                g, dst=dst, op=dist.ReduceOp.SUM,
                group=sp.group, async_op=True,
            )
        
        out = None
        last_work = None
        for i in range(num_parts):
            g = sp.adj_t(i).t() @ grad

            if i > 0:
                last_work.wait()
            
            last_work = async_reduce(i, g)

            if i == part_id:
                out = g

        if last_work is not None:
            last_work.wait()
        return None, out
            
