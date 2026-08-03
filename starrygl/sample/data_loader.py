from collections import deque
from enum import Enum
import pickle
import queue
import threading
import time
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor
import torch
import torch_scatter
import sys
from os.path import abspath, join, dirname
import numpy as np
from starrygl.sample import count_static
from starrygl.sample.batch_data import (
    get_edge_all_to_all_route,
    get_edge_feature_by_dist,
    get_edge_feature_by_dist_async_cached,
    get_node_all_to_all_route,
    get_node_feature_by_dist,
    graph_sample,
    prepare_input,
    prepare_mailbox,
)

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


class _TimedWait:
    """Record only the first, potentially blocking wait on an async work item."""

    def __init__(self, work, recorder, phase):
        self._work = work
        self._recorder = recorder
        self._phase = phase
        self._waited = False

    def wait(self, *args, **kwargs):
        if self._waited:
            return self._work.wait(*args, **kwargs)
        started = time.perf_counter()
        try:
            return self._work.wait(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            self._recorder("fetch_wait", elapsed)
            self._recorder(self._phase, elapsed)
            self._waited = True

    def __getattr__(self, name):
        return getattr(self._work, name)


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
            edge_cache = None,
            edge_communication_cost = 1.0,
            request_planner = None,
            memory_predictor = None,
            edge_cache_collect_signals = True,
            collect_phase_timing = False,
            **kwargs
    ):
        self.ada_param = ada_param
        self.edge_cache = edge_cache
        if edge_communication_cost <= 0:
            raise ValueError("edge_communication_cost must be positive")
        self.edge_communication_cost = float(edge_communication_cost)
        self.request_planner = request_planner
        self.memory_predictor = memory_predictor
        self.edge_cache_collect_signals = bool(edge_cache_collect_signals)
        self.collect_phase_timing = bool(collect_phase_timing)
        self._phase_timing = {}
        self._phase_timing_lock = threading.Lock()
        self.last_request_plan = None
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

    def _record_phase_timing(self, phase, seconds):
        if not self.collect_phase_timing:
            return
        with self._phase_timing_lock:
            seconds_key = '{}_seconds'.format(phase)
            calls_key = '{}_calls'.format(phase)
            self._phase_timing[seconds_key] = (
                self._phase_timing.get(seconds_key, 0.0) + float(seconds)
            )
            self._phase_timing[calls_key] = (
                self._phase_timing.get(calls_key, 0) + 1
            )

    def phase_timing_stats(self):
        if not self.collect_phase_timing:
            return {}
        with self._phase_timing_lock:
            return dict(self._phase_timing)

    def _timed_graph_sample(self, *args, **kwargs):
        if not self.collect_phase_timing:
            return graph_sample(*args, **kwargs)
        started = time.perf_counter()
        try:
            return graph_sample(*args, **kwargs)
        finally:
            self._record_phase_timing(
                'sampler', time.perf_counter() - started
            )

    def _timed_wait(self, work, phase):
        if not self.collect_phase_timing or work is None:
            return work
        return _TimedWait(work, self._record_phase_timing, phase)

    def _edge_cache_signals(self, batch_data, dist_eid):
        if not self.edge_cache_collect_signals:
            return None, None, None
        count = torch.zeros(
            dist_eid.numel(), dtype=torch.float32, device=dist_eid.device
        )
        probability = torch.zeros_like(count)
        variance = torch.zeros_like(count)
        for layer in batch_data[1]:
            for block in layer:
                if '__ID' not in block.edata:
                    continue
                index = block.edata['__ID'].long()
                count += torch.bincount(index, minlength=dist_eid.numel()).float()
                if 'sample_prob' in block.edata:
                    current_probability = block.edata['sample_prob'].reshape(-1).float()
                else:
                    current_probability = torch.ones(
                        index.numel(), dtype=torch.float32, device=index.device
                    )
                layer_probability, _ = torch_scatter.scatter_max(
                    current_probability,
                    index,
                    dim=0,
                    dim_size=dist_eid.numel(),
                )
                probability = torch.maximum(probability, layer_probability)
                if 'residual_variance_gain' in block.edata:
                    variance.index_add_(
                        0,
                        index,
                        block.edata['residual_variance_gain'].reshape(-1).float(),
                    )
        return probability, variance, count

    def _record_request_plan(self, batch_data, dist_nid, dist_eid):
        if self.request_planner is None:
            return
        metadata = batch_data[2]
        cached = (
            self.edge_cache.contains(dist_eid)
            if self.edge_cache is not None
            else torch.zeros(
                dist_eid.numel(), dtype=torch.bool, device=dist_eid.device
            )
        )
        self.last_request_plan = self.request_planner.plan(
            dist_nid,
            dist_eid,
            raw_node_count=metadata.get(
                'core_raw_node_requests', int(dist_nid.numel())
            ),
            raw_edge_count=metadata.get(
                'core_raw_edge_requests', int(dist_eid.numel())
            ),
            edge_cached=cached,
        )

    def _observe_fetched_memory(self, batch_data, dist_nid, memory):
        if self.memory_predictor is None or memory is None:
            return
        metadata = batch_data[2]
        global_nids = metadata.get('core_global_nids')
        if global_nids is None or global_nids.numel() != dist_nid.numel():
            return
        node_memory, memory_ts = memory[0], memory[1]
        owners = DistIndex(dist_nid).part
        remote = owners != dist.get_rank()
        if not bool(remote.any()):
            return
        residual = self.memory_predictor.observe(
            global_nids[remote],
            node_memory[remote],
            memory_ts[remote],
            owner_ids=owners[remote],
        )
        if hasattr(self.sampler, 'observe_core_fetch'):
            self.sampler.observe_core_fetch(
                global_nids[remote],
                residual,
                memory_ts[remote],
            )

    def _observe_edge_state(self, batch_data, dist_eid, edge_feature):
        if (
            edge_feature is None
            or dist_eid.numel() == 0
            or edge_feature.numel() == 0
            or getattr(self.sampler, 'policy', None) != 'boundery_recent_core'
            or not hasattr(self.sampler, 'update_core_edge_state')
        ):
            return
        global_eids = batch_data[2].get('core_global_eids')
        if global_eids is None or global_eids.numel() != dist_eid.numel():
            return
        flattened = edge_feature.detach().reshape(dist_eid.numel(), -1).float()
        edge_error = torch.sqrt(torch.mean(torch.square(flattened), dim=1))
        cached = (
            self.edge_cache.contains(dist_eid)
            if self.edge_cache is not None
            else torch.zeros(
                dist_eid.numel(), dtype=torch.bool, device=dist_eid.device
            )
        )
        edge_error[cached.to(edge_error.device)] = 0.0
        self.sampler.update_core_edge_state(
            global_eids,
            edge_error,
            cached,
        )
              
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
                self._timed_graph_sample,self.graph,self.sampler,
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

        wait_started = time.perf_counter()
        batch_data,dist_nid,dist_eid = self.result_queue[0].result()
        self._record_phase_timing(
            'sampler_future_wait', time.perf_counter() - wait_started
        )
        self._record_request_plan(batch_data, dist_nid, dist_eid)
        self.remote_node += (DistIndex(dist_nid).part != dist.get_rank()).sum().item()
        self.local_node += (DistIndex(dist_nid).part == dist.get_rank()).sum().item()
        self.remote_edge += (DistIndex(dist_eid).part != dist.get_rank()).sum().item()
        self.local_edge += (DistIndex(dist_eid).part == dist.get_rank()).sum().item()
        sync_started = time.perf_counter()
        stream.synchronize()
        self._record_phase_timing(
            'sampler_stream_sync', time.perf_counter() - sync_started
        )
        self.result_queue.popleft()

        route_started = time.perf_counter()
        nind,ndata = get_node_all_to_all_route(self.graph,self.mailbox,dist_nid,out_device=self.device)
        if self.edge_cache is not None and self.graph.efeat is not None:
            edge_probability, edge_variance, edge_reuse = self._edge_cache_signals(
                batch_data, dist_eid
            )
            edge_feat = get_edge_feature_by_dist_async_cached(
                self.graph,
                dist_eid,
                self.edge_cache,
                out_device=self.device,
                access_probability=edge_probability,
                communication_cost=self.edge_communication_cost,
                variance_reduction=edge_variance,
                reuse_count=edge_reuse,
            )
            eind = None
            edata = None
        else:
            eind,edata = get_edge_all_to_all_route(
                self.graph,dist_eid,out_device=self.device
            )

        self._record_phase_timing(
            'all_to_all_route', time.perf_counter() - route_started
        )
        launch_started = time.perf_counter()
        if nind is not None:
            node_feat = DistributedTensor.all_to_all_get_data(ndata,send_ptr=nind['send_ptr'],recv_ptr=nind['recv_ptr'],is_async=True)
            node_feat = (
                node_feat[0],
                self._timed_wait(node_feat[1], 'node_fetch_wait'),
            )
        else:
            node_feat = None

        if self.edge_cache is not None and self.graph.efeat is not None:
            pass
        elif eind is not None:
            edge_feat = DistributedTensor.all_to_all_get_data(edata,send_ptr=eind['send_ptr'],recv_ptr=eind['recv_ptr'],is_async=True)
            edge_feat = (
                edge_feat[0],
                self._timed_wait(edge_feat[1], 'edge_fetch_wait'),
            )
        else:
            edge_feat = None
        self._record_phase_timing(
            'all_to_all_launch', time.perf_counter() - launch_started
        )
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
                batch_data,dist_nid,dist_eid = self._timed_graph_sample(
                    self.graph,
                                      self.sampler,
                                      self.sampler_fn,
                                      data,self.neg_sampler,
                                      self.device,
                                      nid_mapper = self.graph.nids_mapper,
                                      eid_mapper = self.graph.eids_mapper,
                                      reversed = self.reversed
                                      )
                self._record_request_plan(batch_data, dist_nid, dist_eid)

                root,mfgs,metadata = batch_data
                edge_probability, edge_variance, edge_reuse = self._edge_cache_signals(
                    batch_data, dist_eid
                )
                fetch_started = time.perf_counter()
                edge_feat = get_edge_feature_by_dist(
                    self.graph,
                    dist_eid,
                    is_local,
                    out_device=self.device,
                    cache=self.edge_cache,
                    access_probability=edge_probability,
                    communication_cost=self.edge_communication_cost,
                    variance_reduction=edge_variance,
                    reuse_count=edge_reuse,
                )
                node_feat,mem = get_node_feature_by_dist(self.graph,self.mailbox,dist_nid, is_local,out_device=self.device)
                self._record_phase_timing(
                    'bootstrap_fetch', time.perf_counter() - fetch_started
                )
                prepare_input(node_feat,edge_feat,mem,mfgs,dist_nid,dist_eid)
                self._observe_fetched_memory(batch_data, dist_nid, mem)
                self._observe_edge_state(batch_data, dist_eid, edge_feat)
                if(self.mailbox is not None and self.mailbox.historical_cache is not None):
                        id = batch_data[1][0][0].srcdata['ID']
                        mask = DistIndex(id).is_shared
                        batch_data[1][0][0].srcdata['his_mem']  = batch_data[1][0][0].srcdata['mem'].clone()
                        batch_data[1][0][0].srcdata['his_ts']  = batch_data[1][0][0].srcdata['mail_ts'].clone()
                        indx =  self.mailbox.is_shared_mask[DistIndex(batch_data[1][0][0].srcdata['ID']).loc[mask]]
                        batch_data[1][0][0].srcdata['his_mem'][mask] = self.mailbox.historical_cache.local_historical_data[indx]
                        batch_data[1][0][0].srcdata['his_ts'][mask] = self.mailbox.historical_cache.local_ts[indx].reshape(-1,1)
                self.recv_idxs += 1
                assert batch_data is not None
                return root,mfgs,metadata
            else :
                raise StopIteration
        else:
            if self.recv_idxs == 0:
                data = self._next_data()
                batch_data,dist_nid,dist_eid = self._timed_graph_sample(
                    self.graph,
                                      self.sampler,
                                      self.sampler_fn,
                                      data,self.neg_sampler,
                                      self.device,
                                      nid_mapper = self.graph.nids_mapper,
                                      eid_mapper = self.graph.eids_mapper,
                                      reversed = self.reversed
                                    )
                self._record_request_plan(batch_data, dist_nid, dist_eid)
                edge_probability, edge_variance, edge_reuse = self._edge_cache_signals(
                    batch_data, dist_eid
                )
                fetch_started = time.perf_counter()
                edge_feat = get_edge_feature_by_dist(
                    self.graph,
                    dist_eid,
                    is_local,
                    out_device=self.device,
                    cache=self.edge_cache,
                    access_probability=edge_probability,
                    communication_cost=self.edge_communication_cost,
                    variance_reduction=edge_variance,
                    reuse_count=edge_reuse,
                )
                node_feat,mem = get_node_feature_by_dist(self.graph,self.mailbox,dist_nid,  is_local,out_device=self.device)
                self._record_phase_timing(
                    'bootstrap_fetch', time.perf_counter() - fetch_started
                )
                prepare_input(node_feat,edge_feat,mem,batch_data[1],dist_nid,dist_eid)
                self._observe_fetched_memory(batch_data, dist_nid, mem)
                self._observe_edge_state(batch_data, dist_eid, edge_feat)
                if(self.mailbox is not None and self.mailbox.historical_cache is not None):
                    batch_data[1][0][0].srcdata['his_mem']  = batch_data[1][0][0].srcdata['mem'].clone()
                    batch_data[1][0][0].srcdata['his_ts']  = batch_data[1][0][0].srcdata['mail_ts'].clone()
                

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
                            edge_feat[1].wait()
                            edge_feat = edge_feat[0]
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
                        self._record_request_plan(batch_data, dist_nid, dist_eid)
                        stream.synchronize()
                        self.result_queue.popleft()
                        edge_probability, edge_variance, edge_reuse = self._edge_cache_signals(
                            batch_data, dist_eid
                        )
                        edge_feat = get_edge_feature_by_dist(
                            self.graph,
                            dist_eid,
                            is_local,
                            out_device=self.device,
                            cache=self.edge_cache,
                            access_probability=edge_probability,
                            communication_cost=self.edge_communication_cost,
                            variance_reduction=edge_variance,
                            reuse_count=edge_reuse,
                        )
                        node_feat,mem = get_node_feature_by_dist(self.graph,self.mailbox,   dist_nid,is_local,out_device=self.device)
                    prepare_input(node_feat,edge_feat,mem,batch_data[1],dist_nid,dist_eid)
                    self._observe_fetched_memory(batch_data, dist_nid, mem)
                    self._observe_edge_state(batch_data, dist_eid, edge_feat)
                    if(self.mailbox is not None and self.mailbox.historical_cache is not None):
                        id = batch_data[1][0][0].srcdata['ID']
                        mask = DistIndex(id).is_shared
                        batch_data[1][0][0].srcdata['his_mem']  = batch_data[1][0][0].srcdata['mem'].clone()
                        batch_data[1][0][0].srcdata['his_ts']  = batch_data[1][0][0].srcdata['mail_ts'].clone()
                        indx =  self.mailbox.is_shared_mask[DistIndex(batch_data[1][0][0].srcdata['ID']).loc[mask]]
                        batch_data[1][0][0].srcdata['his_mem'][mask] = self.mailbox.historical_cache.local_historical_data[indx]
                        batch_data[1][0][0].srcdata['his_ts'][mask] = self.mailbox.historical_cache.local_ts[indx].reshape(-1,1)
                    
                    self.recv_idxs += 1
                else:
                    raise StopIteration
            global executor
            if(len(self.result_queue)==0):
                self.submit()
        return batch_data
                    
