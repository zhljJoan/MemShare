import torch
import torch.nn as nn

from torch import Tensor
from typing import *

from .buffer import TensorBuffers



class BatchHandle:
    def __init__(self,
        src_ids: Tensor,
        dst_size: int,
        feat_buffer: TensorBuffers,
        grad_buffer: TensorBuffers,
        with_feat0: bool = False,
        layer_id: Optional[int] = None,
        target_device: Any = None,
    ) -> None:
        self.src_ids = src_ids
        self.dst_size = dst_size
        self.feat_buffer = feat_buffer
        self.grad_buffer = grad_buffer
        self.with_feat0 = with_feat0
        self.layer_id = layer_id
        
        if target_device is None:
            self.target_device = src_ids.device
        else:
            self.target_device = torch.device(target_device)

        if with_feat0:
            _ = self.feat0
    
    @property
    def dst_ids(self) -> Tensor:
        return self.src_ids[:self.dst_size]
    
    @property
    def src_size(self) -> int:
        return self.src_ids.size(0)
    
    @property
    def device(self) -> torch.device:
        return self.src_ids.device
    
    @property
    def feat0(self) -> Tensor:
        if not hasattr(self, "_feat0"):
            self._feat0 = self.feat_buffer.get(0, self.src_ids).to(self.target_device)
        return self._feat0
    
    def fetch_feat(self, layer_id: Optional[int] = None) -> Tensor:
        layer_id = int(self.layer_id if layer_id is None else layer_id)
        return self.feat_buffer.get(layer_id, self.src_ids).to(self.target_device)

    def update_feat(self, x: Tensor, layer_id: Optional[int] = None):
        assert x.size(0) == self.dst_size
        layer_id = int(self.layer_id if layer_id is None else layer_id)
        self.feat_buffer.set(layer_id + 1, self.dst_ids, x.detach().to(self.device))
    
    def push_and_pull(self, x: Tensor, layer_id: Optional[int] = None) -> Tensor:
        assert x.size(0) == self.dst_size
        layer_id = int(self.layer_id if layer_id is None else layer_id)
        self.feat_buffer.set(layer_id + 1, self.dst_ids, x.detach().to(self.device))
        o = self.feat_buffer.get(layer_id + 1, self.src_ids[self.dst_size:]).to(x.device)
        return torch.cat([x, o], dim=0)
    
    def fetch_grad(self, layer_id: Optional[int] = None) -> Tensor:
        layer_id = int(self.layer_id if layer_id is None else layer_id)
        return self.grad_buffer.get(layer_id + 1, self.dst_ids).to(self.target_device)
    
    def accumulate_grad(self, x: Tensor, layer_id: Optional[int] = None):
        assert x.size(0) == self.src_size
        layer_id = int(self.layer_id if layer_id is None else layer_id)
        self.grad_buffer.add(layer_id, self.src_ids, x.detach().to(self.device))


class DataLoader:
    def __init__(self,
        node_parts: Tensor,
        feat_buffer: TensorBuffers,
        grad_buffer: TensorBuffers,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_time: Optional[Tensor] = None,
        node_time: Optional[Tensor] = None,
    ) -> None:
        self.src_size = feat_buffer.src_size
        self.dst_size = node_parts.size(0)
        assert node_parts.size(0) <= self.src_size
        assert grad_buffer.src_size == self.src_size

        num_parts = node_parts.max().item() + 1
        cluster, node_perm = node_parts.sort(dim=0)
        node_ptr: Tensor = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

        edge_parts = node_parts[edge_index[1]]
        cluster, edge_perm = edge_parts.sort()
        edge_ptr: Tensor = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)

        self.num_parts = num_parts
        self.node_ptr = node_ptr
        self.edge_ptr = edge_ptr
        self.node_perm = node_perm
        self.edge_perm = edge_perm
        self.edge_index = edge_index[:,edge_perm]

        if edge_attr is not None:
            self.edge_attr = edge_attr[edge_perm]

        if node_time is not None:
            self.node_time = node_time[node_perm]

        if edge_time is not None:
            self.edge_time = edge_time[edge_perm]

        self.feat_buffer = feat_buffer
        self.grad_buffer = grad_buffer
    
    def iter(self, batch_size: int = 1, layer_id: Optional[int] = None, seed: int = 0, filter: Callable[[Tensor], Tensor] = None, device = None):
        rnd = torch.Generator()
        if seed != 0:
            rnd.manual_seed(seed)
        sampled = torch.randperm(self.num_parts, generator=rnd, dtype=torch.long)
        
        s = 0
        imp = torch.empty(self.src_size, dtype=torch.long)
        while s < sampled.size(0):
            t = min(s + batch_size, sampled.size(0))

            dst_ids = []
            edge_index = []
            edge_attr = []

            imp.zero_()
            for i in sampled[s:t].tolist():
                s += batch_size

                a, b = self.node_ptr[i:i+2].tolist()
                nidx = self.node_perm[a:b]
                if hasattr(self, "node_time") and filter is not None:
                    node_mask = filter(self.node_time[a:b])
                    if not node_mask.any():
                        continue
                    nidx = nidx[node_mask]
                else:
                    node_mask = None
                
                a, b = self.edge_ptr[i:i+2].tolist()
                eidx = self.edge_index[:, a:b]
                if hasattr(self, "edge_time") and filter is not None:
                    if node_mask is None:
                        edge_mask = filter(self.edge_time[a:b])
                    else:
                        imp[nidx] = i+1
                        edge_mask = (imp[eidx[1]] == i+1)
                        edge_mask &= filter(self.edge_time[a:b])
                    if not edge_mask.any():
                        continue
                    eidx = eidx[:, edge_mask]
                else:
                    edge_mask = None
                
                dst_ids.append(nidx)
                edge_index.append(eidx)

                if hasattr(self, "edge_attr"):
                    attr = self.edge_attr[a:b]
                    if edge_mask is None:
                        edge_attr.append(attr)
                    else:
                        edge_attr.append(attr[edge_mask])

            if len(dst_ids) == 0 or len(edge_index) == 0:
                continue
            dst_ids = torch.cat(dst_ids, dim=-1)
            edge_index = torch.cat(edge_index, dim=-1)
            if hasattr(self, "edge_attr"):
                edge_attr = torch.cat(edge_attr, dim=0)
            else:
                edge_attr = None

            imp.zero_()
            imp.index_fill_(0, edge_index[0], 1).index_fill_(0, dst_ids, 0)
            src_ids = torch.cat([dst_ids, torch.where(imp > 0)[0]], dim=-1)
            assert (src_ids[:dst_ids.size(0)] == dst_ids).all()

            imp.fill_((2**62-1)*2+1)
            imp[src_ids] = torch.arange(src_ids.size(0), dtype=torch.long)
            edge_index = imp[edge_index.flatten()].view_as(edge_index)

            handle = BatchHandle(
                src_ids, dst_ids.size(0),
                self.feat_buffer, self.grad_buffer,
                with_feat0 = (layer_id is None),
                layer_id = layer_id,
                target_device = device,
            )
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)

            yield handle, edge_index, edge_attr
