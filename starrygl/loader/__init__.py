import torch
import torch.nn as nn

from torch import Tensor
from typing import *

from torch_sparse import SparseTensor
from torch_scatter import scatter_min

from multiprocessing.pool import ThreadPool
from .buffer import TensorBuffer
from .route import Route
from .context import RouteContext

# class BatchHandle:
#     def __init__(self,
#         src_ids: Tensor,
#         dst_size: int,
#         feat_buffer: TensorBuffers,
#         grad_buffer: TensorBuffers,
#         with_feat0: bool = False,
#         layer_id: Optional[int] = None,
#         target_device: Any = None,
#     ) -> None:
#         self.src_ids = src_ids
#         self.dst_size = dst_size
#         self.feat_buffer = feat_buffer
#         self.grad_buffer = grad_buffer
#         self.with_feat0 = with_feat0
#         self.layer_id = layer_id
        
#         if target_device is None:
#             self.target_device = src_ids.device
#         else:
#             self.target_device = torch.device(target_device)

#         if with_feat0:
#             _ = self.feat0
    
#     @property
#     def dst_ids(self) -> Tensor:
#         return self.src_ids[:self.dst_size]
    
#     @property
#     def src_size(self) -> int:
#         return self.src_ids.size(0)
    
#     @property
#     def device(self) -> torch.device:
#         return self.src_ids.device
    
#     @property
#     def feat0(self) -> Tensor:
#         if not hasattr(self, "_feat0"):
#             self._feat0 = self.feat_buffer.get(0, self.src_ids).to(self.target_device)
#         return self._feat0
    
#     def fetch_feat(self, layer_id: Optional[int] = None) -> Tensor:
#         layer_id = int(self.layer_id if layer_id is None else layer_id)
#         return self.feat_buffer.get(layer_id, self.src_ids).to(self.target_device)

#     def update_feat(self, x: Tensor, layer_id: Optional[int] = None):
#         assert x.size(0) == self.dst_size
#         layer_id = int(self.layer_id if layer_id is None else layer_id)
#         self.feat_buffer.set(layer_id + 1, self.dst_ids, x.detach().to(self.device))
    
#     def push_and_pull(self, x: Tensor, layer_id: Optional[int] = None) -> Tensor:
#         assert x.size(0) == self.dst_size
#         layer_id = int(self.layer_id if layer_id is None else layer_id)
#         self.feat_buffer.set(layer_id + 1, self.dst_ids, x.detach().to(self.device))
#         o = self.feat_buffer.get(layer_id + 1, self.src_ids[self.dst_size:]).to(x.device)
#         return torch.cat([x, o], dim=0)
    
#     def fetch_grad(self, layer_id: Optional[int] = None) -> Tensor:
#         layer_id = int(self.layer_id if layer_id is None else layer_id)
#         return self.grad_buffer.get(layer_id + 1, self.dst_ids).to(self.target_device)
    
#     def accumulate_grad(self, x: Tensor, layer_id: Optional[int] = None):
#         assert x.size(0) == self.src_size
#         layer_id = int(self.layer_id if layer_id is None else layer_id)
#         self.grad_buffer.add(layer_id, self.src_ids, x.detach().to(self.device))

class NodeHandle:
    def __init__(self,
        src_ids: Tensor,
        dst_size: int,
        edge_ids: Tensor,
        adj_t: SparseTensor,
        executor: ThreadPool,
        device: Any,
        stream: Optional[torch.cuda.Stream] = None,
        edge_time: Optional[Tensor] = None,
        node_time: Optional[Tensor] = None,
    ) -> None:
        self._src_ids = src_ids
        self._dst_size = dst_size
        self._edge_ids = edge_ids

        self._device = torch.device(device)
        if self._device != torch.device("cpu") and stream is None:
            self._stream = torch.cuda.Stream(self._device)
        else:
            self._stream = stream
        self._executor = executor

        self._adj_t = adj_t.to(self.device)
        self._node_time = node_time
        self._edge_time = edge_time
    
    def get_src_feats(self, data: Union[Tensor, TensorBuffer]):
        return self.async_select(data, self.src_ids)

    def get_dst_feats(self, data: Union[Tensor, TensorBuffer]):
        return self.async_select(data, self.dst_ids)

    def get_ext_feats(self, data: Union[Tensor, TensorBuffer]):
        return self.async_select(data, self.ext_ids)

    def get_edge_feats(self, data: Union[Tensor, TensorBuffer]):
        return self.async_select(data, self.edge_ids)

    def get_src_time(self):
        return self.async_select(self._node_time, self.src_ids)

    def get_dst_time(self):
        return self.async_select(self._node_time, self.dst_ids)
    
    def get_ext_time(self):
        return self.async_select(self._node_time, self.ext_ids)

    def get_edge_time(self) -> Tensor:
        return self.async_select(self._edge_time, self.edge_ids)
    
    def push_and_pull(self, value: Tensor, ext_fut: torch.futures.Future[Tensor], data: Union[Tensor, TensorBuffer]):
        fut = self.async_update(data, value, self.dst_ids)
        ext_value = ext_fut.wait()
        x = torch.cat([value, ext_value], dim=0)
        return x, fut

    def async_select(self, src: Union[Tensor, TensorBuffer], idx: Tensor) -> torch.futures.Future[Tensor]:
        fut = torch.futures.Future(devices=[self._device])
        def run():
            try:
                with torch.no_grad():
                    with torch.cuda.stream(self._stream):
                        index = idx.to(src.device)
                        if isinstance(src, TensorBuffer):
                            val = src.local_get(index, lock=True)
                        else:
                            val = src.index_select(0, index)
                        val = val.to(self._device)
                        fut.set_result(val)
            except Exception as e:
                fut.set_exception(e)
        self._executor.apply_async(run)
        return fut
    
    def async_update(self, src: Union[Tensor, TensorBuffer], val: Tensor, idx: Tensor, ops: str = "mov") -> torch.futures.Future:
        assert ops in ["mov", "add"]
        fut = torch.futures.Future(devices=[self._device])
        def run():
            try:
                with torch.no_grad():
                    with torch.cuda.stream(self._stream):
                        value = val.to(src.device)
                        index = idx.to(src.device)
                        if isinstance(src, TensorBuffer):
                            if ops == "mov":
                                src.all_remote_set(value, index, lock=True)
                            elif ops == "add":
                                src.all_remote_add(value, index, lock=True)
                        else:
                            if ops == "mov":
                                src[index] = value
                            elif ops == "add":
                                src[index] += value
                        fut.set_result(None)
            except Exception as e:
                fut.set_exception(e)
        self._executor.apply_async(run)
        return fut
    
    @property
    def adj_t(self) -> SparseTensor:
        return self._adj_t
    
    @property
    def src_ids(self) -> Tensor:
        return self._src_ids
    
    @property
    def src_size(self) -> int:
        return self._src_ids.size(0)
    
    @property
    def dst_ids(self) -> Tensor:
        return self._src_ids[:self._dst_size]
    
    @property
    def dst_size(self) -> int:
        return self._dst_size
    
    @property
    def ext_ids(self) -> Tensor:
        return self._src_ids[self._dst_size:]
    
    @property
    def ext_size(self) -> int:
        return self.src_size - self.dst_size
    
    @property
    def edge_ids(self) -> Tensor:
        return self._edge_ids
    
    @property
    def edge_size(self) -> Tensor:
        return self._edge_ids.size(0)
    
    @property
    def device(self) -> torch.device:
        return self._device

class NodeLoader:
    def __init__(self,
        global_ids: Tensor,
        global_edges: Tensor,
        device: Any,
        edge_time: Optional[Tensor] = None,
        node_time: Optional[Tensor] = None,
        num_threads: int = 1,
    ) -> None:
        self.route, edge_index = Route.from_edge_index(global_ids, global_edges)

        if node_time is None and edge_time is not None:
                node_time = scatter_min(edge_time[edge_index[0]], edge_index[1], dim=0, dim_size=self.dst_size)
        
        node_ids = torch.arange(self.dst_size).type_as(global_ids)
        if node_time is not None:
            perm = node_time.argsort()
            node_ids = node_ids[perm]

        self.node_ids = node_ids
        self.node_time = node_time

        edge_ids = torch.arange(edge_index.size(1)).type_as(global_ids)
        perm = edge_index[1].argsort()

        edge_ids = edge_ids[perm]
        edge_index = edge_index[:,perm]

        self.edge_ids = edge_ids
        self.edge_index = edge_index
        self.edge_time = edge_time
        # self.rowptr = torch.ops.torch_sparse.ind2ptr(self.edge_index[1], self.dst_size)

        self.device = torch.device(device)
        if self.device != torch.device("cpu"):
            self.stream = torch.cuda.Stream(self.device)
        else:
            self.stream = None
        self.executor = ThreadPool(num_threads)
    
    @property
    def src_size(self) -> int:
        return self.route.src_size
    
    @property
    def dst_size(self) -> int:
        return self.route.dst_size
    
    # def _select_nodes(self, start_time: int, end_time: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    #     if self.node_time is None:
    #         if self.edge_time is None:
    #             raise RuntimeError("neither node_time nor edge_time exists!")
    #         else:
    #             edge_mask = (start_time <= self.edge_time) & (self.edge_time < end_time)
    #             idx = self.edge_index[1, edge_mask]
    #             node_mask = torch.zeros(self.dst_size, dtype=torch.bool, device=idx.device).index_fill_(0, idx, 1)
    #             return torch.where(node_mask)[0], None
    #     else:
    #         time_range = torch.tensor([start_time, end_time]).type_as(self.node_time)
    #         s, t = torch.searchsorted(self.node_time, time_range).tolist()
    #         return self.node_ids[s:t], self.node_time[s:t]
    
    def _get_node_handle(self, node_ids: Tensor, edge_ids: Tensor, edge_index: Tensor) -> NodeHandle:
        num_node: int = torch.max(node_ids.max(), edge_index.max()).item() + 1
        imp = torch.zeros(num_node, dtype=torch.bool, device=node_ids.device)
        imp.index_fill_(0, edge_index[0], 1)
        imp.index_fill_(0, node_ids, 0)
        src_ids = torch.cat([node_ids, torch.where(imp)[0]], dim=0)
        src_size = src_ids.size(0)
        dst_size = node_ids.size(0)

        imp = torch.zeros(num_node, dtype=torch.long, device=node_ids.device).fill_((2**62-1)*2+1)
        imp[src_ids] = torch.arange(src_size, dtype=torch.long, device=node_ids.device)
        edge_index = imp[edge_index.flatten()].view_as(edge_index)

        perm = edge_index[1].argsort()
        edge_ids = edge_ids[perm]
        edge_index = edge_index[:,perm]

        rowptr = torch.ops.torch_sparse.ind2ptr(edge_index[1], dst_size)
        adj_t = SparseTensor(rowptr=rowptr, col=edge_index[0], sparse_sizes=(dst_size, src_size))
        return NodeHandle(
            src_ids=src_ids,
            dst_size=dst_size,
            edge_ids=edge_ids,
            adj_t=adj_t,
            executor=self.executor,
            device=self.device,
            stream=self.stream,
            edge_time=self.edge_time,
            node_time=self.node_time,
        )
    
    # very very very slow !!!
    def iter(self, batch_size: int, wind: bool = False, start_time: int = 0, end_time: int = 9223372036854775807, seed: int = 0):
        rand_gen = torch.Generator()
        if seed != 0:
            rand_gen.manual_seed(seed)
        
        if not wind:
            sampled = torch.randperm(self.dst_size, generator=rand_gen, dtype=torch.long)
            for s in range(0, sampled.size(0), batch_size):
                t = min(s + batch_size, sampled.size(0))
                node_ids = sampled[s:t]
                imp = torch.zeros(self.dst_size, dtype=torch.bool, device=node_ids.device)
                imp.index_fill_(0, node_ids, 1)
                edge_mask = imp[self.edge_index[1]]
                edge_ids = self.edge_ids[edge_mask]
                edge_index = self.edge_index[:,edge_mask]
                yield self._get_node_handle(node_ids, edge_ids, edge_index)
        else:
            assert self.node_time is not None
            node_mask = (start_time <= self.node_time) & (self.node_time < end_time)
            all_node_ids = torch.where(node_mask)[0]

            sampled = torch.randperm(node_ids.size(0), generator=rand_gen, dtype=torch.long)
            all_node_ids = all_node_ids[sampled]

            for s in range(0, sampled.size(0), batch_size):
                t = min(s + batch_size, sampled.size(0))
                node_ids = all_node_ids[s:t]
                
                imp = torch.zeros(self.dst_size, dtype=torch.bool, device=node_ids.device)
                imp.index_fill_(0, node_ids, 1)
                edge_mask = imp[self.edge_index[1]]
                if self.edge_time is not None:
                    edge_time = self.edge_time[self.edge_ids]
                    edge_mask &= (start_time <= edge_time) & (edge_time < end_time)
                edge_ids = self.edge_ids[edge_mask]
                edge_index = self.edge_index[:,edge_mask]
                yield self._get_node_handle(node_ids, edge_ids, edge_index)

