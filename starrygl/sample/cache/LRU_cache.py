from typing import Optional, Sequence, Union

import torch
from starrygl.distributed.utils import DistributedTensor

from starrygl.sample.cache.cache import Cache


class LRUCache(Cache):
    """
    Least-recently-used (LRU) cache
    """

    def __init__(self,  cache_ratio: int,
                 num_cache:int,
                 cache_data: Sequence[DistributedTensor],
                 use_local:bool = False,
                 pinned_buffers_shape: Sequence[torch.Size] = None,
                 is_update_cache = False
                 
            ):
        super(LRUCache, self).__init__(cache_ratio,num_cache,
                 cache_data,use_local,
                 pinned_buffers_shape,
                 is_update_cache)
        self.name = 'lru'
        self.now_cache_count = 0
        self.cache_count = torch.zeros(
                self.capacity, dtype=torch.int32, device=torch.device('cuda'))
        self.is_update_cache = True


    def update_cache(self, cached_index: torch.Tensor,
                          uncached_index: torch.Tensor,
                          uncached_feature: Sequence[torch.Tensor]):
        
        if len(uncached_index) > self.capacity:
            num_to_cache = self.capacity
        else:
            num_to_cache = len(uncached_index)
        node_id_to_cache = uncached_index[:num_to_cache].to(torch.int32)
        self.now_cache_count -= 1
        self.cache_count[cached_index] = 0

        # get the k node id with the least water level
        removing_cache_index = torch.topk(
            self.cache_count, k=num_to_cache, largest=False).indices.to(torch.int32)
    
        removing_node_id = self.cache_index_to_id[removing_cache_index]

        # update cache attributes
        for buffer,data in zip(self.buffers,uncached_feature):
            
            buffer[removing_cache_index] = data[:num_to_cache].reshape(-1,*buffer.shape[1:])
        self.cache_count[removing_cache_index] = 0
        self.cache_validate[removing_node_id] = False
        self.cache_validate[node_id_to_cache] = True
        self.cache_map[removing_node_id] = -1
        self.cache_map[node_id_to_cache] = removing_cache_index
        self.cache_index_to_id[removing_cache_index] = node_id_to_cache
