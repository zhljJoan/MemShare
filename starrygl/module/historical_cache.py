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


    def __init__(self,cache_index,layer,shape,dtype,device,threshold = 3,time_threshold = None, times_threshold = 10, use_rpc = True, num_threshold = 0):
        #self.cache_index = cache_index
        self.layer = layer
        print(shape)
        #self.data = torch.zeros(cache_index.historical_num,shape,dtype = dtype, device= torch.device('cpu'))
        self.local_historical_data = torch.zeros(cache_index.shape[0],shape,dtype = dtype, device= device)
        #print(self.data.shape)
        #self.ts = torch.zeros(cache_index.historical_num,dtype = torch.float,device = torch.device('cpu'))
        self.local_ts = torch.zeros(cache_index.shape[0],dtype = torch.float,device = device)
        self.loss_count = torch.zeros(cache_index.shape[0],dtype = torch.int,device = device)
        self.threshold = threshold
        self.time_threshold = time_threshold
        self.times_threshold = times_threshold
        self.num_threshold = num_threshold
        self._ctx = DistributedContext.get_default_context()
        self.use_rpc = use_rpc
        #self.pinned_memory = torch.empty_like(self.data,device=torch.device('cpu')).pin_memory()
        self.last_shared_update_wait = None
        #if use_rpc:
        #    self.rref = rpc.RRef(self)
        #    self.rrefs = self._ctx.all_gather_remote_objects(self.rref)
        #self.update_wait_list = None#[deque() for _ in range(len(self.rrefs))]
        #self.update_cnt = 0
        #self.broadcast_cnt = 0
        #self.update_thread = None
        #self.now_cnt = 0
        #self.get_cnt = [0 for _ in range(len(self.rrefs))]
        #self.buffer = None
        #self.buffer_index = None
        #self.pinned_buffer = torch.
    def empty(self):
        #self.data.zero_()
        #self.ts.zero_()
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
        if self.time_threshold is not None:
            mask = (self.ssim(new_data,self.local_historical_data[index]) > self.threshold | (ts - self.local_ts[index] > self.time_threshold | self.loss_count[index] > self.times_threshold))
            self.loss_count[index][~mask] += 1
            self.loss_count[index][mask] = 0
        else:
            #print('{} {} {} {}  \n'.format(index,self.ssim(new_data,self.local_historical_data[index]),new_data,self.local_historical_data[index]))
            #print(new_data,self.local_historical_data[index])
            #print(self.ssim(new_data,self.local_historical_data[index]) < self.threshold, (self.loss_count[index] > self.times_threshold))
            mask = (self.ssim(new_data,self.local_historical_data[index]) > self.threshold) | (self.loss_count[index] > self.times_threshold)
            self.loss_count[index][~mask] += 1
            self.loss_count[index][mask] = 0
        return mask
    
    def read_synchronize(self):
        get_stream_set(self.layer).synchronize()
    
    # def get_cold_data(self,index,to_device):
    #     with get_stream_set(self.layer):
    #         index = index.to('cpu')
    #         index = self.cache_index.cache_index[index]
    #         pin_hist = _get_pin(pinn_memory, self.layer, index.shape[0], self.data[1:])
    #         torch.index_select(self.data,0,index,pin_hist)
    #         return pin_hist.to(to_device,non_blocking = True),self.ts[index].to(to_device)
    
    # def get_hot_data(self,index):
    #     return self.local_historical_data[index]

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
        shared_data = torch.cat(shared_data,dim = 0)
        shared_index = torch.cat(shared_index)
        if(shared_data.shape[0] == 0):
            return None
        len = self.local_historical_data.shape[1]
        #mail_ts = shared_data[:,-1]
        #mail_data = shared_data[:,len+1:-1]
        shared_ts = shared_data[:,len]
        shared_mem = shared_data[:,:len]
        #print(shared_index)
        unq_index,inv = torch.unique(shared_index,return_inverse = True)
        max_ts,idx = torch_scatter.scatter_max(shared_ts,inv,0)
        #shared_ts =  torch_scatter.scatter_mean(shared_ts,inv,0)
        #shared_data =  torch_scatter.scatter_mean(shared_data,inv,0)
        shared_mem = shared_mem[idx]
        shared_ts = shared_ts[idx]
        #mail_data = mail_data[idx]
        #mail_ts = mail_ts[idx]
        shared_index = unq_index
        #print('{} {} {}\n'.format(shared_index,shared_data,shared_ts))
        # if filter is not None:
        #     change = shared_data - self.local_historical_data[shared_index] 
        #     if not (change.max().item() == 0):     
        #         change -= change.min()
        #         change /=change.max() 
        #         change = 2*change - 1
        #         filter.update(shared_index,change)
        self.local_historical_data[shared_index] = shared_mem
        self.local_ts[shared_index] = shared_ts
        self.last_shared_update_wait = None
        return shared_index,shared_mem,shared_ts#,mail_data,mail_ts
        

        
    def add_shared_to_queue(self,handle0,handle1,shared_id_list,shared_data):
        self.last_shared_update_wait = (handle0,handle1,shared_id_list,shared_data)

    # def _update(self,_index, origin_id, new_data,new_ts):
    #     with get_stream_set(self.layer):
    #         _,ind = _index.unique(return_inverse=True)
    #         max_ts,ind = torch_scatter.scatter_max(ts,ind)
    #         ts = max_ts
    #         _index = _index[ind]
    #         new_data = new_data[ind]
    #         origin_id = origin_id[ind]
    #         index = DistIndex(_index).loc
    #         self.loss_count[index] = self.loss_count[index] + 1
    #         idx = self.historical_check(index,new_data,ts)
    #         if idx.sum().item() > 0:
    #             index = index[idx] 
    #             origin_id = origin_id[idx]
    #             new_data = new_data[idx]
    #             ts = ts[idx]
    #             self.loss_count[index]=0
    #             self.local_historical_data[index] = new_data
    #             self.ts[index] = index
    #             self.broadcast(index[idx], origin_id[idx], new_data[idx],ts[idx])
        
    
    # def update(self,_index,origin_id,new_data,ts):
    #     #while(self.update_thread is not None and self.update_thread.done() is False):
    #     #    pass
    #     self.update_thread.result()
    #     self.update_thread = self.update_thread = executor.submit(self._update,_index,origin_id,new_data,ts)
    #     #self._update(_index,origin_id,new_data,ts)
        
    # #def async_update()
    # def set_cache(self,_index,data,ts,src_rank):
    #     local_stream = get_stream_set(self.layer)
    #     with torch.cuda.stream(local_stream):
    #         index = self.cache_index.cache_index[index]
    #         index = index.contiguous().to(self.data.device)
    #         data = data.contiguous().to(self.data.device)
    #         ts = ts.contiguous().to(self.data.device)
    #         self.data[index] = data.to(self.data.device)
    #         self.ts[index] = ts.to(self.ts.device).to(self.ts.dtype)

    # @staticmethod
    # def update_remote_cache(rref,index,data,ts):
    #     rref.rpc_async().set_cache(index,data,ts,dist.get_rank())
        
    # def _broadcast(self,index,origin_id,data,ts):
    #     ctx = DistributedContext.get_default_context()
    #     data = data.contiguous().to('cpu')
    #     ts = ts.contiguous().to('cpu')
    #     for r in range(ctx.memory_group_size):
    #         if r == ctx.memory_group_rank:
    #             continue
    #         mask = (self.cache_index.route[index] & (1<<r))>0
    #         self.update_remote_cache(self.rrefs[r],origin_id[mask],data[mask],ts[mask])   

    # def broadcast(self,index,origin_id,data,ts):

    #     self._broadcast(index,origin_id,data,ts)
    #     #executor.submit(self._broadcast,index,data,ts) 
            
