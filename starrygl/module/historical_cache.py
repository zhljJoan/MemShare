from collections import deque
import threading
from typing import Optional, Sequence, Union

import torch
import torch_scatter
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor

from starrygl.sample.cache.cache import Cache
import torch.distributed.rpc as rpc
import torch.distributed as dist
import concurrent.futures

from starrygl.sample.graph_core.utils import _get_pin

class CachePushRoute():
    def __init__(self,local_num, local_drop_edge, dist_index):
        ctx = DistributedContext.get_default_context()
        dist_drop = dist_index[local_drop_edge]
        id_mask = not (DistIndex(dist_drop) == ctx.memory_group_rank)
        remote_id = local_drop_edge[id_mask]
        route = torch.zeros(local_num,dtype=torch.long)
        src_mask = DistIndex(dist_drop[0,:]).part == ctx.memory_group_rank
        route[DistIndex(dist_drop[0,src_mask]).loc] |= (1<<DistIndex(dist_drop[1,src_mask]).part)
        dst_mask = DistIndex(dist_drop[1,:]).part == ctx.memory_group_rank
        route[DistIndex(dist_drop[0,dst_mask]).loc] |= (1<<DistIndex(dist_drop[1,dst_mask]).part)
        self.route = route
        self.cache_index = torch.empty_like(dist_index)
        self.cache_index[remote_id] = torch.arange(remote_id.shape[0],device=remote_id.device)
        self.historical_num = remote_id.shape[0]
        

stream_set = {}
def get_stream_set(layer):
    if layer in stream_set is False:
        stream_set[layer] = torch.cuda.Stream()
    return stream_set[layer]
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
pinn_memory = {}
class HistoricalCache:


    def __init__(self,cache_index,layer,shape,dtype,device,ada_param = None,time_threshold = None, times_threshold = 10, use_rpc = True, num_threshold = 0):
        #self.cache_index = cache_index
        self.layer = layer
        print(shape)
        self.local_historical_data = torch.zeros(cache_index.shape[0],shape,dtype = dtype, device= device)
        self.local_ts = torch.zeros(cache_index.shape[0],dtype = torch.float,device = device)
        self.loss_count = torch.zeros(cache_index.shape[0],dtype = torch.int,device = device)
        self.threshold = ada_param.alpha
        self.ada_param = ada_param
        self.time_threshold = time_threshold
        self.times_threshold = times_threshold
        self.num_threshold = num_threshold
        self._ctx = DistributedContext.get_default_context()
        self.use_rpc = use_rpc
        self.last_shared_update_wait = None
        
    def empty(self):
        self.loss_count.zero_()
        self.now_cnt = 0
        self.local_historical_data.zero_()
        self.local_ts.zero_()
        self.loss_count.zero_()

    def ssim(self,x,y,type = 'cos'):
        
        if type == 'cos':
            return 1-torch.nn.functional.cosine_similarity(x,y) 
        else: 
            return torch.sum((x -y)**2,dim = 1)
    def historical_check(self,index,new_data,ts):
        self.threshold = self.ada_param.alpha
        if self.time_threshold is not None:
            mask = (self.ssim(new_data,self.local_historical_data[index]) > self.threshold | (ts - self.local_ts[index] > self.time_threshold | self.loss_count[index] > self.times_threshold))
            self.loss_count[index][~mask] += 1
           
            self.loss_count[index][mask] = 0
        else:
            mask = ((self.ssim(new_data,self.local_historical_data[index]) > self.threshold) | (self.loss_count[index] > self.times_threshold))
            (self.loss_count[index[~mask]]) += 1
            self.loss_count[index[mask]] = 0
        return mask
    
    def read_synchronize(self):
        get_stream_set(self.layer).synchronize()
    


    def get_shared_update_buffer(self,shape0,shape1,device,dtype):
        buffer = self.buffer
        if(buffer is None):
            buffer = []
            for l in shape0:
                buffer = torch.empty([shape0[0],shape1],device = device,dtype = dtype)
        else:
            for i,l in enumerate(shape0):
                if(buffer[i] < l[0]):
                    buffer[i].resize_(l)
        shared_list = [buffer[:l[0]] for l in shape0]
        return shared_list
    
    def synchronize_shared_update(self,filter=None):
        if self.last_shared_update_wait is None:
            return None
        handle0,handle1,shared_index,shared_data = self.last_shared_update_wait
        self.last_shared_update_wait = None
        handle0.wait()
        handle1.wait()
        if self.ada_param.training is True:
            self.ada_param.update_memory_sync_time(self.ada_param.last_start_event_memory_sync)
        shared_data = torch.cat(shared_data,dim = 0)
        shared_index = torch.cat(shared_index)
        if(shared_data.shape[0] == 0):
            return None
        len = self.local_historical_data.shape[1]

        shared_ts = shared_data[:,len]
        shared_mem = shared_data[:,:len]
        unq_index,inv = torch.unique(shared_index,return_inverse = True)
        max_ts,idx = torch_scatter.scatter_max(shared_ts,inv,0)
        shared_mem = shared_mem[idx]
        shared_ts = shared_ts[idx]
        shared_index = unq_index
        self.local_historical_data[shared_index] = shared_mem
        self.local_ts[shared_index] = shared_ts
        self.last_shared_update_wait = None
        return shared_index,shared_mem,shared_ts#,mail_data,mail_ts
        

        
    def add_shared_to_queue(self,handle0,handle1,shared_id_list,shared_data):
        self.last_shared_update_wait = (handle0,handle1,shared_id_list,shared_data)