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
    We will perform feature fetch in the data loader.
    you can simply define a data loader for use, while starrygl assisting in fetching node or edge features:

        
        
    Args:
        graph: distributed graph store
    
        data: the graph data
        
        sampler: a parallel sampler like `NeighborSampler` above
        
        sampler_fn: sample type
        
        neg_sampler: negative sampler
        
        batch_size: batch size
        
        mailbox: APAN's mailbox and TGN's memory implemented by starrygl
    
    Examples:

        .. code-block:: python
        
            import torch

            from starrygl.sample.data_loader import DistributedDataLoader
            from starrygl.sample.part_utils.partition_tgnn import partition_load
            from starrygl.sample.graph_core import DataSet, DistributedGraphStore, TemporalNeighborSampleGraph
            from starrygl.sample.memory.shared_mailbox import SharedMailBox
            from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler
            from starrygl.sample.sample_core.base import NegativeSampling
            from starrygl.sample.batch_data import SAMPLE_TYPE

            pdata = partition_load("PATH/{}".format(dataname), algo="metis_for_tgnn")    
            graph = DistributedGraphStore(pdata = pdata, uvm_edge = False, uvm_node = False)
            sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
            mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat=pdata.edge_attr.shape[1] if pdata.  edge_attr is not None else 0)
            sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=1, fanout=[10], graph_data=sample_graph,    workers=15,policy = 'recent',graph_name = "wiki_train")
            neg_sampler = NegativeSampling('triplet')
            train_data = torch.masked_select(graph.edge_index, pdata.train_mask.to(graph.edge_index.device)).reshape    (2, -1)
            trainloader = DistributedDataLoader(graph, train_data, sampler=sampler, sampler_fn=SAMPLE_TYPE. SAMPLE_FROM_TEMPORAL_EDGES,neg_sampler=neg_sampler, batch_size=1000, shuffle=False, drop_last=True, chunk_size = None,train=True, mailbox=mailbox )

    In the data loader, we will call the `graph_sample`, sourced from `starrygl.sample.batch_data`.

    And the `to_block` function in the `graph_sample` will implement feature fetching.
    If cache is not used, we will directly fetch node or edge features from the graph data, 
    otherwise we will call `fetch_data` for feature fetching.     

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
            is_pipeline = False,
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
        self.is_pipeline = is_pipeline
        if train is True:
            self._get_expected_idx(self.dataset.len)
        else:
            self._get_expected_idx(self.dataset.len,op = dist.ReduceOp.MAX)
            #self.expected_idx = int(math.ceil(self.dataset.len/self.batch_size))
        torch.distributed.barrier()
              
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
            num_batchs = torch.tensor([self.expected_idx],dtype = torch.long,device=self.device) 
            print("num_batchs:", num_batchs)
            dist.all_reduce(num_batchs, op=op)
            self.expected_idx = int(num_batchs.item())

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
        if self.is_pipeline is False:
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
                torch.cuda.synchronize()
                return batch_data
            else :
                raise StopIteration
        else:
            if self.recv_idxs == 0:
                data = self._next_data()
                batch_data = graph_sample(self.graph,
                                          self.sampler,
                                          self.sampler_fn,
                                          data,self.neg_sampler,
                                          self.mailbox,
                                          self.device)
                self.recv_idxs += 1
            else:
                if(self.recv_idxs < self.expected_idx):
                    assert len(self.result_queue) > 0
                    result= self.result_queue[0]
                    self.result_queue.popleft()
                    batch_data = result.result()
                    self.recv_idxs += 1
                else:
                    raise StopIteration
            if(self.recv_idxs+1<=self.expected_idx):
                data = self._next_data()
                next_batch =  graph_sample(self.graph,
                                          self.sampler,
                                          self.sampler_fn,
                                          data,self.neg_sampler,
                                          self.mailbox,
                                          self.device,
                                          async_op=True)
                self.result_queue.append(next_batch)     
        return batch_data
                    

        
    


    
        
        
        
