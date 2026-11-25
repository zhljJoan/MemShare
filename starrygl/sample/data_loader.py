from collections import deque
from enum import Enum
import os
import pickle
import queue
import threading
import time
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor
import torch
import sys
from os.path import abspath, join, dirname
import numpy as np
from starrygl.sample import count_static
from starrygl.sample.batch_data import get_edge_all_to_all_route, get_edge_feature_by_dist, get_node_all_to_all_route, get_node_feature_by_dist, graph_sample, prepare_input,prepare_mailbox

from starrygl.sample.count_static import time_count
from starrygl.sample.part_utils.transformer_from_speed import get_eval_batch
from starrygl.sample.sample_core.PreNegSampling import PreNegativeSampling

sys.path.insert(0, join(abspath(dirname(__file__))))
from typing import Deque, Optional
import torch.distributed as dist
from torch_geometric.data import Data
import os.path as osp
import math
from starrygl.sample.count_static import time_count as tt
import data_loader as data_loader
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) 
stream = torch.cuda.Stream()


def _prepare_historical_input(block, mailbox, refresh_shared):
    if (
        os.environ.get('MEMSHARE_COMPACT_MEMORY_OPS') == 'on'
        and hasattr(block, '_memshare_memory_object')
        and hasattr(block, '_memshare_dist_nid_object')
    ):
        historical_memory = block._memshare_memory_object.clone()
        if refresh_shared:
            object_ids = block._memshare_dist_nid_object
            shared_mask = DistIndex(object_ids).is_shared
            shared_index = mailbox.is_shared_mask[
                DistIndex(object_ids[shared_mask]).loc]
            historical_memory[shared_mask] = (
                mailbox.historical_cache.local_historical_data[shared_index])
        block._memshare_historical_memory_object = historical_memory
        return

    block.srcdata['his_mem'] = block.srcdata['mem'].clone()
    block.srcdata['his_ts'] = block.srcdata['mail_ts'].clone()
    if refresh_shared:
        shared_mask = DistIndex(block.srcdata['ID']).is_shared
        shared_index = mailbox.is_shared_mask[
            DistIndex(block.srcdata['ID']).loc[shared_mask]]
        block.srcdata['his_mem'][shared_mask] = (
            mailbox.historical_cache.local_historical_data[shared_index])
        block.srcdata['his_ts'][shared_mask] = (
            mailbox.historical_cache.local_ts[shared_index].reshape(-1, 1))


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
            mode = 'train',
            queue_size = 10,
            mailbox = None,
            is_pipeline = False,
            local_embedding = False,
            cache_mask = None,
            use_local_feature = True,
            probability = 1,
            reversed = False,
            ada_param = None,
            **kwargs
    ):
        self.ada_param = ada_param
        self.reversed = reversed
        self.use_local_feature = use_local_feature
        self.local_embedding = local_embedding
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
        self.is_train = (mode == 'train')
        self.previous_edges = None
        if self.is_train and os.getenv("MEMSHARE_PREVIOUS_EDGE_FEATURES") == "1" and graph.efeat is not None:
            from starrygl.sample.previous_edge_features import PreviousEdgeFeatures
            self.previous_edges = PreviousEdgeFeatures(
                sorted_ids=not getattr(sampler, "hierarchical_compaction", False))
        self.mode = mode
        self.cache_mask = cache_mask
        #if self.is_train is True:
        #    self._get_expected_idx(self.dataset.len)
        #else:
        batch_pos_l, batch_pos_r, pos_id = get_eval_batch(dataset.eids,graph.nids_mapper,graph.eids_mapper,self.batch_size)
        self.batch_pos_l = batch_pos_l
        self.batch_pos_r = batch_pos_r
        self.dataset = self.dataset[pos_id]
        diff_count = torch.tensor([(DistIndex(graph.nids_mapper[self.dataset.edges[0,:].to('cpu')]).part!=DistIndex(graph.nids_mapper[self.dataset.edges[1,:].to('cpu')]).part).sum()])
        ctx = DistributedContext.get_default_context()
        dist.all_reduce(diff_count,group = ctx.gloo_group)
        print('all_cross_edge:{}  local edges num: {}\n'.format(diff_count,(DistIndex(graph.eids_mapper[dataset.eids.to('cpu')]).part == dist.get_rank()).sum()))
        self.expected_idx = batch_pos_r.size(0)
        torch.distributed.barrier()
        self.local_node = 0
        self.remote_node = 0
        self.local_edge = 0
        self.remote_edge = 0
        self.local_root = 0
        self.remote_root = 0
        self.local_root = 0
        self.probability = probability
        print('pro {}\n'.format(self.probability))
              
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

        if self.batch_pos_l[self.submitted] == -1:
            next_data = self.input_dataset[torch.tensor([],device=self.device,dtype= torch.long)]
        else:
            next_data = self.input_dataset[self.batch_pos_l[self.submitted]:self.batch_pos_r[self.submitted] +1]
        if self.mode=='train' and self.ada_param.beta < 1:
            mask = ((DistIndex(self.graph.nids_mapper[next_data.edges[0,:].to('cpu')]).part == dist.get_rank())&(DistIndex(self.graph.nids_mapper[next_data.edges[1,:].to('cpu')]).part == dist.get_rank()))
            if self.ada_param.beta > 0:
                mask[~mask] = (torch.rand((~mask).sum().item()) < self.ada_param.beta)
            next_data = next_data[mask.to(next_data.device)]
        self.submitted = self.submitted + 1
        return next_data
    def submit(self):
        global executor
        if(self.submitted < self.expected_idx):
            data = self._next_data()  
            with torch.cuda.stream(stream):
                fut = executor.submit(
                graph_sample,self.graph,self.sampler,
                            self.sampler_fn,
                            data,self.neg_sampler,
                            self.device,
                            nid_mapper = self.graph.nids_mapper,
                            eid_mapper = self.graph.eids_mapper,
                            reversed = self.reversed
                                      
                )

                self.result_queue.append((fut))
    @torch.no_grad()
    def async_feature(self):
        if(self.recv_idxs >= self.expected_idx or self.is_pipeline == False):
            return
        is_local = (self.is_train & self.use_local_feature)
        if(is_local):
            return
        while(len(self.result_queue)==0):
            pass

        batch_data,dist_nid,dist_eid = self.result_queue[0].result()
        self.remote_node += (DistIndex(dist_nid).part != dist.get_rank()).sum().item()
        self.local_node += (DistIndex(dist_nid).part == dist.get_rank()).sum().item()
        self.remote_edge += (DistIndex(dist_eid).part != dist.get_rank()).sum().item()
        self.local_edge += (DistIndex(dist_eid).part == dist.get_rank()).sum().item()
        stream.synchronize()
        self.result_queue.popleft()
        
        nind,ndata = get_node_all_to_all_route(self.graph,self.mailbox,dist_nid,out_device=self.device)
        requested, plan = (self.previous_edges.lookup(dist_eid) if self.previous_edges is not None else (dist_eid, None))
        eind,edata = get_edge_all_to_all_route(self.graph,requested,out_device=self.device)
        
        if nind is not None:
            node_feat = DistributedTensor.all_to_all_get_data(ndata,send_ptr=nind['send_ptr'],recv_ptr=nind['recv_ptr'],is_async=True)
        else:
            node_feat = None

        if eind or self.reversed is False:
            edge_feat = DistributedTensor.all_to_all_get_data(edata,send_ptr=eind['send_ptr'],recv_ptr=eind['recv_ptr'],is_async=True)
        else:
            edge_feat = None
        if self.previous_edges is not None and edge_feat is not None:
            edge_feat = (*edge_feat, plan)
        self.result_queue.append((batch_data,dist_nid,dist_eid,edge_feat,node_feat))
        self.submit()
        if self.ada_param is not None:
           
            self.ada_param.last_start_event_fetch = self.ada_param.start_event()

    @torch.no_grad()
    def __next__(self):
        ctx = DistributedContext.get_default_context()
        is_local = (self.is_train & self.use_local_feature)
        t0 = time.time()
        #with torch.cuda.stream(stream):
        if self.is_pipeline is False:
            if self.recv_idxs < self.expected_idx:
                t0 = tt.start_gpu()
                data = self._next_data()
                batch_data,dist_nid,dist_eid = graph_sample(
                    self.graph,
                                      self.sampler,
                                      self.sampler_fn,
                                      data,self.neg_sampler,
                                      self.device,
                                      nid_mapper = self.graph.nids_mapper,
                                      eid_mapper = self.graph.eids_mapper,
                                      reversed = self.reversed
                                      )

                root,mfgs,metadata = batch_data
                edge_feat = get_edge_feature_by_dist(self.graph,dist_eid,is_local,out_device=self.device)
                node_feat,mem = get_node_feature_by_dist(self.graph,self.mailbox,dist_nid, is_local,out_device=self.device)
                prepare_input(node_feat,edge_feat,mem,mfgs,dist_nid,dist_eid)
                if(self.mailbox is not None and self.mailbox.historical_cache is not None):
                        _prepare_historical_input(
                            batch_data[1][0][0], self.mailbox, True)
                self.recv_idxs += 1
                assert batch_data is not None
                return root,mfgs,metadata
            else :
                raise StopIteration
        else:
            if self.recv_idxs == 0:
                data = self._next_data()
                batch_data,dist_nid,dist_eid = graph_sample(
                    self.graph,
                                      self.sampler,
                                      self.sampler_fn,
                                      data,self.neg_sampler,
                                      self.device,
                                      nid_mapper = self.graph.nids_mapper,
                                      eid_mapper = self.graph.eids_mapper,
                                      reversed = self.reversed
                                    )
                edge_feat = get_edge_feature_by_dist(self.graph,dist_eid,is_local,out_device=self.device)
                node_feat,mem = get_node_feature_by_dist(self.graph,self.mailbox,dist_nid,  is_local,out_device=self.device)
                if self.previous_edges is not None and edge_feat is not None:
                    self.previous_edges.complete(dist_eid, edge_feat, None)
                prepare_input(node_feat,edge_feat,mem,batch_data[1],dist_nid,dist_eid)
                if(self.mailbox is not None and self.mailbox.historical_cache is not None):
                    _prepare_historical_input(
                        batch_data[1][0][0], self.mailbox, False)
                

                self.recv_idxs += 1
            else:
                
                if(self.recv_idxs < self.expected_idx):
                    assert len(self.result_queue) > 0
                    #print(len(self.result_queue[0]))
                    if isinstance(self.result_queue[0],tuple) :
                        
                        t0 = time.time()
                        batch_data,dist_nid,dist_eid,edge_feat,node_feat0 = self.result_queue[0]
                        self.result_queue.popleft()

                        if edge_feat is not None:
                            plan = edge_feat[2] if self.previous_edges is not None else None
                            edge_feat[1].wait()
                            edge_feat = edge_feat[0]
                            if self.previous_edges is not None:
                                edge_feat = self.previous_edges.complete(dist_eid, edge_feat, plan)
                                if os.getenv("MEMSHARE_VERIFY_EDGE_CACHE") == "1":
                                    ref = get_edge_feature_by_dist(self.graph, dist_eid, False, out_device=self.device)
                                    assert torch.equal(ref, edge_feat)
                                    rank = dist.get_rank()
                                    remote = DistIndex(dist_eid).part != rank
                                    hits = plan[1] if plan is not None else dist_eid[:0]
                                    path = os.path.join(os.environ["MEMSHARE_CACHE_TRACE_DIR"], f"edge-{rank}.tsv")
                                    with open(path, "a") as out:
                                        print(len(dist_eid), len(hits), remote.sum().item(), remote[hits].sum().item(), file=out)
                        node_feat = None
                        mem = None
                        if self.graph.nfeat is not None:
                            node_feat0[1].wait()
                            node_feat0 = node_feat0[0]
                            node_feat = node_feat0[:,:self.graph.nfeat.shape[1]]
                            if self.graph.nfeat.shape[1] < node_feat0.shape[1]:
                                mem = self.mailbox.unpack(node_feat0[:,self.graph.nfeat.shape[1]:],mailbox = True)
                            else:
                                mem = None
                        elif self.mailbox is not None:
                            node_feat0[1].wait()
                            node_feat0 = node_feat0[0]
                            node_feat = None
                            mem = self.mailbox.unpack(node_feat0,mailbox = True)
                    else:
                        batch_data,dist_nid,dist_eid = self.result_queue[0].result()
                        stream.synchronize()
                        self.result_queue.popleft()
                        edge_feat = get_edge_feature_by_dist(self.graph,dist_eid,is_local,out_device=self.device)
                        node_feat,mem = get_node_feature_by_dist(self.graph,self.mailbox,   dist_nid,is_local,out_device=self.device)
                    prepare_input(node_feat,edge_feat,mem,batch_data[1],dist_nid,dist_eid)
                    if(self.mailbox is not None and self.mailbox.historical_cache is not None):
                        _prepare_historical_input(
                            batch_data[1][0][0], self.mailbox, True)
                    
                    self.recv_idxs += 1
                else:
                    raise StopIteration
            global executor
            if(len(self.result_queue)==0):
                self.submit()
        return batch_data
                    
