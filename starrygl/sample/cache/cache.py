from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from starrygl.distributed.utils import DistributedTensor



class Cache:
    def __init__(self,  cache_ratio: int,
                 num_cache:int,
                 cache_data: Sequence[DistributedTensor],
                 use_local:bool = False,
                 pinned_buffers_shape: Sequence[torch.Size] = None,
                 is_update_cache = False
                 
            ):
        print(len(cache_data),cache_data)
        assert torch.cuda.is_available() == True
        self.use_local = use_local
        self.is_update_cache = is_update_cache
        self.device = torch.device('cuda')
        self.use_remote = torch.distributed.get_world_size()>1
        assert not (self.use_local is False and self.use_remote is False),\
            "the data is on the cuda and no need remote cache"
        self.cache_ratio = cache_ratio
        self.num_cache = num_cache
        self.capacity = int(self.num_cache * cache_ratio)
        self.update_stream = torch.cuda.Stream()
        self.buffers = []
        self.pinned_buffers = []
        for data in cache_data:
            self.buffers.append(
                torch.zeros(self.capacity,*data.shape[1:],
                            dtype = data.dtype,device = torch.device('cuda'))
            )
        self.cache_validate = torch.zeros(
            num_cache, dtype=torch.bool, device=self.device)
        # maps node id -> index
        self.cache_map = torch.zeros(
            num_cache, dtype=torch.int32, device=self.device) - 1
        # maps index -> node id
        self.cache_index_to_id = torch.zeros(
            num_cache,dtype=torch.int32, device=self.device) -1
        self.hit_sum = 0
        self.hit_ = 0
    def init_cache(self,ind:torch.Tensor,data:Sequence[torch.Tensor]):
        pos = torch.arange(ind.shape[0],device = 'cuda',dtype = ind.dtype)
        self.cache_map[ind] = pos.to(torch.int32).to('cuda')
        self.cache_index_to_id[pos] = ind.to(torch.int32).to('cuda')
        for data,buffer in zip(data,self.buffers):
            buffer[:ind.shape[0],] = data
        self.cache_validate[ind] = True
    
    def update_cache(self, cached_index: torch.Tensor,
                          uncached_index: torch.Tensor,
                          uncached_data: Sequence[torch.Tensor]):
        raise NotImplementedError
    
    def fetch_data(self,ind:Optional[torch.Tensor] = None,
                    uncached_source_fn: Callable = None, source_index:torch.Tensor = None):
        self.hit_sum += ind.shape[0]
        
        assert isinstance(ind, torch.Tensor)
        cache_mask = self.cache_validate[ind]
        uncached_mask = ~cache_mask
        self.hit_ += torch.sum(cache_mask) 
        cached_data = []
        cached_index = self.cache_map[ind[cache_mask]]
        if uncached_mask.sum() > 0:
            uncached_id = ind[uncached_mask]
            source_index = source_index[uncached_mask]
            uncached_feature = uncached_source_fn(source_index)
            if isinstance(uncached_feature,torch.Tensor):
                uncached_feature = [uncached_feature]
        else:
            uncached_id = None
            uncached_feature = [None for _ in range(len(self.buffers))]
        for data,uncached_data in zip(self.buffers,uncached_feature): 
            nfeature = torch.zeros(
                len(ind), *data.shape[1:], dtype=data.dtype,device=self.device)
            nfeature[cache_mask,:] = data[cached_index]
            if uncached_id is not None:
                nfeature[uncached_mask] = uncached_data.reshape(-1,*data.shape[1:])
            cached_data.append(nfeature)
        if self.is_update_cache and uncached_mask.sum() > 0:
            self.update_cache(cached_index=cached_index,
                                   uncached_index=uncached_id,
                                   uncached_feature=uncached_feature)
        return nfeature
    
    def invalidate(self,ind):
        self.cache_validate[ind] = False
 
    

