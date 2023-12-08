# import torch
# import torch.nn as nn

# from torch.utils import hooks

# from torch import Tensor
# from typing import *

# from .route import Route, RouteWorkBase
# from .shrink import ShrinkData
# from .acopy import get_executor as get_acopy_executor, AsyncCopyWorkBase
# from .route import get_route_executor


# class ProbeLayer:
#     def __init__(self, layer_id, probe_obj) -> None:
#         self.layer_id: int = layer_id
#         self.probe_obj: NodeProbe = probe_obj
#         self.last_w = torch.ones(self.probe_obj.num_nodes, dtype=torch.float32)
#         self._val_hook_handle: Optional[hooks.RemovableHandle] = None
    
#     def to(self, device):
#         self.last_w = self.last_w.to(device)
#         return self
    
#     def remove_val_hook(self):
#         if self._val_hook_handle is not None:
#             self._val_hook_handle.remove()
#         self._val_hook_handle = None
    
#     def register_val_hook(self, val: Tensor, idx: Optional[Tensor]):
#         assert self._val_hook_handle is None, "cannot call register_val_hook() twice"

#         def hook(grad: Tensor):
#             from starrygl.utils.printer import main_print
#             import time
#             self.probe_obj.update_sample_w(self.last_w, grad, idx)
#             self.remove_val_hook()
#             self._backward_sample(False)
#             main_print(self.layer_id, grad.size(0), idx.size(0))

#         self.val_hook_handle = val.register_hook(hook)
    
#     def warmup_sample(self):
#         assert self._is_last_layer()
#         for probe_layer in self.probe_obj.layers[::-1]:
#             probe_layer._backward_sample()
            
#             # max_norm = max(probe_layer.last_w.max().item(), 1.0)
#             # probe_layer.last_w[probe_layer.last_w == 1.0] = max_norm
    
#     def _backward_sample(self, x: bool = True):
#         dst_idx = self._collect_prev_dst_idx()
#         if dst_idx is None:
#             return
        
#         for cache in self.probe_obj.hook_caches:
#             next_cache_layer = self._next_cache_layer(cache)
#             shrink = next_cache_layer.set_shrink_data(dst_idx)

#             src_idx = shrink.src_idx

#             work = get_acopy_executor().async_pull(next_cache_layer.data, src_idx)
#             next_cache_layer.sync_pull_work(work)
            
#             if not self._is_first_layer() and x:
#                 next_cache_layer.sync_backward_sample_work(src_idx)

#     def _collect_prev_dst_idx(self) -> Optional[Tensor]:
#         if len(self.probe_obj.hook_caches) == 0:
#             return None
        
#         if self._is_last_layer():
#             return self._wrapped_sample(self.probe_obj.num_samples, None)
        
#         if len(self.probe_obj.hook_caches) == 1:
#             for cache in self.probe_obj.hook_caches:
#                 prev_cache_layer = self._prev_cache_layer(cache)
#                 dst_idx = prev_cache_layer.sync_backward_sample_work()
#         else:
#             tmp = torch.zeros(self.probe_obj.num_nodes, dtype=torch.bool, device=self.last_w.device)
#             for cache in self.probe_obj.hook_caches:
#                 prev_cache_layer = self._prev_cache_layer(cache)
#                 prev_idx = prev_cache_layer.sync_backward_sample_work()
#                 if prev_idx is not None:
#                     tmp.index_fill_(0, prev_idx, 1)
#             dst_idx = torch.where(tmp)[0]
#             if dst_idx.size(0) == 0:
#                 dst_idx = None

#         if dst_idx is None:
#             return None
#         return self._wrapped_sample(self.probe_obj.num_samples, dst_idx)

#     def _prev_cache_layer(self, cache):
#         cache: MessageCache = cache
#         i = self.layer_id + 1
#         if i < self.probe_obj.num_layers:
#             return cache.layers[i]
#         return None

#     def _next_cache_layer(self, cache):
#         cache: MessageCache = cache
#         return cache.layers[self.layer_id]
    
#     def _is_last_layer(self):
#         return self.layer_id + 1 >= self.probe_obj.num_layers
    
#     def _is_first_layer(self):
#         return self.layer_id == 0
    
#     def _wrapped_sample(self, k: int, idx: Optional[Tensor]) -> Optional[Tensor]:
#         w = self.last_w
#         if idx is None:
#             return self.probe_obj.sample(w, k) if 0 < k and k < self.probe_obj.num_nodes else None
#         if 0 < k and k < idx.size(0):
#             t = self.probe_obj.sample(w[idx], k)
#             return idx[t]
#         else:
#             return idx

# class NodeProbe:
#     def __init__(self,
#         num_nodes: int,
#         num_layers: int,
#         num_samples: int = 0,
#         p: str = "fro",
#         dim: int = -1,
#         beta: float = 1.0,
#     ) -> None:
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.num_layers = num_layers
#         self.num_samples = num_samples
#         self.p = p
#         self.dim = dim
#         self.beta = beta

#         self.layers = [ProbeLayer(i, self) for i in range(num_layers)]
#         self.hook_caches: Set[MessageCache] = set()
    
#     def to(self, device):
#         for i in range(self.num_layers):
#             self.layers[i].to(device)
#         return self
    
#     def assign_message_cache(self, cache):
#         cache: MessageCache = cache
#         assert self.num_layers == cache.num_layers
#         self.hook_caches.add(cache)
    
#     def update_sample_w(self, last_w: Tensor, grad: Tensor, idx: Optional[Tensor]):
#         val_norm = grad.norm(p=self.p, dim=self.dim)
#         if idx is None:
#             last_w[:] = val_norm
#         else:    
#             if self.beta != 1.0:
#                 last_w.mul_(self.beta)
#             last_w[idx] = val_norm
    
#     def sample(self, w: Tensor, k: int) -> Optional[Tensor]:
#         w = w / w.sum()
#         return torch.multinomial(w, num_samples=k, replacement=False)
    
#     def apply(self, i: int, val: Tensor, idx: Optional[Tensor]) -> Tensor:
#         self.layers[i].register_val_hook(val, idx)
#         return val

# class CacheLayer:
#     def __init__(self, layer_id, cache_obj, num_features) -> None:
#         self.layer_id: int = layer_id
#         self.cache_obj: MessageCache = cache_obj
#         self.data: Tensor = torch.randn(
#             size=(self.cache_obj.src_size, num_features),
#             dtype=torch.float32,
#             device=self.cache_obj.cache_device,
#         )
#         self.state: Dict[str, Any] = {}
#         self.shrink: Optional[ShrinkData] = None
#         self._async_push_work: Optional[AsyncCopyWorkBase] = None
#         self._async_pull_work: Optional[AsyncCopyWorkBase] = None
#         self._backward_sample_work: Optional[RouteWorkBase] = None
    
#     def to(self, device):
#         if self.shrink is not None:
#             self.shrink = self.shrink.to(device)
#         return self
    
#     def cache_data_to(self):
#         self.data = self.data.to(self.cache_obj.cache_device)
#         return self
    
#     def set_shrink_data(self, dst_idx: Optional[Tensor]) -> Optional[ShrinkData]:
#         self.shrink = self.cache_obj.new_shrink_data(dst_idx)
#         return self.shrink
    
#     def get_shrink_data(self) -> Optional[ShrinkData]:
#         return self.shrink
    
#     # def set_backward_sample_work(self, src_idx: Optional[Tensor]):
#     #     if src_idx is None:
#     #         return
#     #     assert self._backward_sample_work is None
#     #     self._backward_sample_work = self.cache_obj.route.backward_a2a(src_idx, src_idx)

#     # def pop_backward_sample_work(self) -> Optional[RouteWorkBase]:
#     #     work = self._backward_sample_work
#     #     self._backward_sample_work = None
#     #     return work
    
#     def sync_backward_sample_work(self, src_idx: Optional[Tensor] = None) -> Optional[Tensor]:
#         dst_idx = None
#         if self._backward_sample_work is not None:
#             _, dst_idx = self._backward_sample_work.get()

#         if src_idx is not None:
#             # self._backward_sample_work = self.cache_obj.route.backward_a2a(src_idx, src_idx)
#             work = get_route_executor().async_backward_a2a(src_idx, src_idx, self.cache_obj.route)
#             self._backward_sample_work = work
#         else:
#             self._backward_sample_work = None
#         return dst_idx

#     def sync_push_work(self, work: Optional[AsyncCopyWorkBase] = None):
#         if self._async_push_work is not None:
#             self._async_push_work.get()
#         self._async_push_work = work
    
#     def sync_pull_work(self, work: Optional[AsyncCopyWorkBase] = None) -> Optional[Tensor]:
#         out = None
#         if self._async_pull_work is not None:
#             out = self._async_pull_work.get()
#         self._async_pull_work = work
#         return out

# class MessageCache:
#     def __init__(self,
#         src_ids: Tensor,
#         dst_ids: Tensor,
#         edge_index: Tensor,
#         num_features: Union[int, torch.Size],
#         num_layers: int,
#         cache_device: Union[str, torch.device, None],
#         bipartite: bool = False,
#     ) -> None:
#         self.src_size = src_ids.size(0)
#         self.dst_size = dst_ids.size(0)
#         self.edge_index = edge_index
#         self.num_layers = num_layers

#         if cache_device is None:
#             cache_device = edge_index.device
#         self.cache_device = cache_device
#         self.bipartite = bipartite

#         self.route = Route(dst_ids, src_ids, bipartite=bipartite)
#         self.layers = [CacheLayer(i, self, num_features) for i in range(num_layers)]
    
#     def to(self, device):
#         self.edge_index = self.edge_index.to(device)
#         self.route = self.route.to(device)
#         for i in range(self.num_layers):
#             self.layers[i].to(device)
#         return self
    
#     def cached_data_to(self, device):
#         self.cache_device = torch.device(device)
#         for i in range(self.num_layers):
#             self.layers[i].cache_data_to()
#         return self
    
#     # @property
#     # def is_offload(self) -> bool:
#     #     return self.edge_index.device != self.cache_device
    
#     def new_shrink_data(self, dst_idx: Optional[Tensor]) -> Optional[ShrinkData]:
#         if dst_idx is None:
#             return None
#         return ShrinkData(
#             src_size=self.src_size,
#             dst_size=self.dst_size,
#             dst_idx=dst_idx,
#             edge_index=self.edge_index,
#             bipartite=self.bipartite,
#         )
    
#     # def clear_shrink_data(self):
#     #     for i in range(self.num_layers):
#     #         self.layers[i].shrink = None

#     def replace_layer_data(self, i: int, data: Tensor):
#         layer = self.layers[i]
#         assert layer.data.size(0) == data.size(0)

#         layer.data = data
#         return self
    
#     def update_cache(self,
#         i: int,
#         val: Tensor,
#         idx: Optional[Tensor],
#         async_op: bool = False,
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         layer: CacheLayer = self.layers[i]
#         src_val, src_idx = self.route.apply(val, idx, layer.state, async_op=async_op)

#         # full graph
#         if idx is None:
#             return src_val, None
        
#         shrink = layer.get_shrink_data()
#         if shrink is None: # 这里可能存在问题，在初始化的时候
#             data = torch.empty_like(layer.data, device=val.device).copy_(layer.data)
#             data.index_copy_(0, src_idx, src_val)
#             return data, src_idx
#         else:
#             push_work = get_acopy_executor().async_push(layer.data, src_idx, src_val)
#             layer.sync_push_work(push_work)

#             data = layer.sync_pull_work()
#             sval, sidx = shrink.shrink_src_val_and_idx(src_val, src_idx)
#             data.index_copy_(0, sidx, sval)
#             return data, sidx

#         # 先异步更新缓存
#         # 同时异步下载缓存
#         # 然后本地和远程缓存混合，作为最终结果返回
    
#     def fetch_pull_tensor(self, i: int) -> Optional[Tensor]:
#         return self.layers[i].sync_pull_work()
    
#     # def _update_cache_impl(self,
#     #     i: int,
#     #     dst_val: Tensor,
#     #     dst_idx: Optional[Tensor],
#     #     route: Route,
#     #     async_op: bool,
#     # ):
#     #     # communcation
#     #     state = self.layers[i].state
#     #     src_val, src_idx = route.apply(dst_val, dst_idx, state, async_op=async_op)
        
#     #     # push latest embeddings
#     #     data = self.layers[i].data
#     #     if src_idx is None:
#     #         data[:] = src_val
#     #     else:
#     #         data[src_idx] = src_val
        
#     #     # get previous generated shrink data
#     #     shr: ShrinkData = self.layers[i].shrink
#     #     if shr is None:
#     #         return src_val, src_idx
#     #     else:
#     #         # pull latest embeddings
#     #         return data[shr.src_idx], shr.src_idx
    
#     # def _update_cache_offload(self,
#     #     i: int,
#     #     dst_val: Tensor,
#     #     dst_idx: Optional[Tensor],
#     #     route: Route,
#     #     async_op: bool,
#     # ):
#     #     raise NotImplementedError
    
#     # def _compute_cached_data_size(self) -> torch.Size:
#     #     num_features = self.num_features
#     #     if num_features is None:
#     #         cached_data_size = torch.Size([self.src_size,])
#     #     else:
#     #         cached_data_size = torch.Size([self.src_size, num_features])
#     #     return cached_data_size
    
#     # def _compute_cpu_buf_size(self) -> torch.Size:
#     #     num_features = self.num_features
#     #     if num_features is None:
#     #         cpu_buf_size = torch.Size([2**32,])
#     #     else:
#     #         cpu_buf_size = torch.Size([2**32 // num_features, num_features])
#     #     return cpu_buf_size