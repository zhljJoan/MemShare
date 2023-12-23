from collections import deque
from enum import Enum
import queue
import torch
import sys
from os.path import abspath, join, dirname
import numpy as np
from starrygl.sample.batch_data import graph_sample

from starrygl.sample.sample_core.PreNegSampling import PreNegativeSampling

sys.path.insert(0, join(abspath(dirname(__file__))))
from typing import Deque, Optional
import torch.distributed as dist
from torch_geometric.data import Data
import os.path as osp
import math


    

class DistributedDataLoader:
    ''' 
     Args:
            data_path: the path of loaded graph ,each part 0 of graph is saved on $path$/rank_0
            num_replicas: the num of worker
            

    '''
    def __init__(
            self,
            graph,
            dataset = None,
            sampler = None,
            sampler_fn = None,
            neg_sampler = None,
            batch_size: Optional[int]=None,
            drop_last = False,
            device: torch.device  = torch.device('cuda'),
            shuffle:bool = True,
            chunk_size = None,
            train = False,
            queue_size = 10,
            mailbox = None,
            **kwargs
    ):
        assert sampler is not None
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.num_pending = 0
        self.current_pos = 0
        self.recv_idxs = 0
        self.drop_last = drop_last
        self.result_queue = deque(maxlen = self.queue_size)
        self.shuffle = shuffle
        self.is_closed = False
        self.sampler = sampler
        self.sampler_fn = sampler_fn
        self.neg_sampler = neg_sampler
        self.graph = graph
        self.shuffle=shuffle
        self.dataset = dataset
        self.mailbox = mailbox
        self.device =  device
        if train is True:
            self._get_expected_idx(self.dataset.len)
        else:
            self._get_expected_idx(self.dataset.len,op = dist.ReduceOp.MAX)
            #self.expected_idx = int(math.ceil(self.dataset.len/self.batch_size))
              
    def __iter__(self):
        if self.chunk_size is None:
            if self.shuffle:
                self.input_dataset = self.dataset.shuffle()
            else:
                self.input_dataset = self.dataset
            self.recv_idxs = 0
            self.current_pos = 0
            self.num_pending = 0
            self.submitted = 0
        else:
            self.input_dataset = self.dataset
            self.recv_idxs = 0
            self.num_pending = 0
            self.submitted = 0
            if dist.get_rank == 0:
                self.current_pos = int(
                    math.floor(
                        np.random.uniform(0,self.batch_size/self.chunk_size)
                    )*self.chunk_size
                )
            else:
                self.current_pos = 0
            current_pos = torch.tensor([self.current_pos],dtype = torch.long,device=self.device) 
            dist.broadcast(current_pos, src = 0)
            self.current_pos = int(current_pos.item())
            self._get_expected_idx(self.dataset.len-self.current_pos)


        if self.neg_sampler is not None \
            and isinstance(self.neg_sampler,PreNegativeSampling):
            self.neg_sampler.set_next_pos(self.current_pos)
        return self

    def _get_expected_idx(self,data_size,op = dist.ReduceOp.MIN):
        world_size = dist.get_world_size()
        self.expected_idx = data_size // self.batch_size if self.drop_last is True else int(math.ceil(data_size/self.batch_size))

        if dist.get_world_size() > 1:
            num_epochs = torch.tensor([self.expected_idx],dtype = torch.long,device=self.device) 
            print(num_epochs)
            dist.all_reduce(num_epochs, op=op)
            self.expected_idx = int(num_epochs.item())

    def _next_data(self):   
        if self.current_pos >= self.dataset.len:
            return self.input_dataset._get_empty()

        if self.current_pos + self.batch_size > self.input_dataset.len:
            if self.drop_last:
                return None
            else:
                next_data = self.input_dataset.get_next(
                    slice(self.current_pos,None,None)
                )
                self.current_pos = 0
        else:
            next_data = self.input_dataset.get_next(
                slice(self.current_pos,self.current_pos + self.batch_size,None)
            )
            self.current_pos += self.batch_size

        return next_data
    
    def __next__(self):
        if(dist.get_world_size() > 0):
            if self.recv_idxs < self.expected_idx:
                data = self._next_data()
                batch_data = graph_sample(self.graph,
                                          self.sampler,
                                          self.sampler_fn,
                                          data,self.neg_sampler,
                                          self.mailbox,
                                          self.device)
                self.recv_idxs += 1
                assert batch_data is not None
                return batch_data
            else :
                raise StopIteration
        if self.queue_size > 0 :
            num_reqs = min(self.queue_size - self.num_pending,self.expected_idx - self.submitted)
            for _ in range(num_reqs):
                if len(self.result_queue)>0:
                    result = self.result_queue[0]
                    if result[1].done() == True:
                        self.recv_idxs += 1
                        self.num_pending -= 1
                        self.result_queue.popleft()
                        return result[0],result[1].value(),result[2]
                else:
                    next_data = self._next_data()
                    assert next_data is not None
                    batch_data = graph_sample(self.graph,
                                          self.sampler,
                                          self.sampler_fn,
                                          next_data,self.neg_sampler,
                                          self.mailbox,
                                          self.device)
                    batch_data[1].wait()
                    self.submitted = self.submitted + 1
                    self.num_pending = self.num_pending + 1
                    self.recv_idxs += 1
                    self.num_pending -= 1
                    return batch_data[0],batch_data[1].value(),batch_data[2]
                    self.result_queue.append(batch_data)
                    self.submitted = self.submitted + 1
                    self.num_pending = self.num_pending + 1
                    
            while(self.recv_idxs < self.expected_idx):
                assert len(self.result_queue) > 0
                result= self.result_queue[0]
                if result[1].done() == True:
                    self.recv_idxs += 1
                    self.num_pending -= 1
                    self.result_queue.popleft()
                    return result[0],result[1].value(),result[2]
                
                
            assert self.num_pending == 0
            raise StopIteration
                    

        
    


    
        
        
        
