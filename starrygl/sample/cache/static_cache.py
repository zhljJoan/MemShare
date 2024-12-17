from typing import Optional, Sequence, Union

import torch
from starrygl.distributed.utils import DistributedTensor

from starrygl.sample.cache.cache import Cache


class StaticCache(Cache):


    def __init__(self,  cache_ratio: int,
                 num_cache:int,
                 cache_data: Sequence[DistributedTensor],
                 use_local:bool = False,
                 pinned_buffers_shape: Sequence[torch.Size] = None,
                 is_update_cache = False
                 
            ):
        super(StaticCache, self).__init__(cache_ratio,num_cache,
                 cache_data,use_local,
                 pinned_buffers_shape,
                 is_update_cache)
        self.name = 'static'
        self.now_cache_count = 0
        self.cache_count = torch.zeros(
                self.capacity, dtype=torch.int32, device=torch.device('cuda'))
        self.is_update_cache = False

