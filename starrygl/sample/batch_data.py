import time
from typing import List, Tuple
import torch
import torch.distributed as dist
import torch_scatter
from starrygl import distributed
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor
from starrygl.sample import count_static
from starrygl.sample.cache.fetch_cache import FetchFeatureCache
from starrygl.sample.count_static import time_count
from starrygl.sample.graph_core import DataSet
from starrygl.sample.graph_core import DistributedGraphStore
from starrygl.sample.memory.shared_mailbox import SharedMailBox
from starrygl.sample.sample_core.base import BaseSampler, NegativeSampling
import dgl
from starrygl.sample.count_static import time_count as tt
from starrygl.sample.stream_manager import PipelineManager, getPipelineManger
"""
入参不变，出参变为：
sample_from_nodes
node: list[tensor,tensor, tensor...]
eid: list[tensor,tensor, tensor...]
src_index: list[tensor,tensor, tensor...]

sample_from_edges：
node
eid: list[tensor,tensor, tensor...]
src_index: list[tensor,tensor, tensor...]
delta_ts: list[tensor,tensor, tensor...]
metadata
"""
class CountComm:
    origin_remote = 0
    origin_local = 0
    def __init__(self):
        pass
total_sample_core_time = 0
total_fetch_prepare_time = 0
total_comm_time = 0
total_build_time = 0
total_prepare_input_time = 0
total_build_block_time = 0
def set_zero():
    global total_sample_core_time
    total_sample_core_time = 0
    global total_fetch_prepare_time
    total_fetch_prepare_time = 0
    global total_comm_time
    total_comm_time = 0
    global total_build_time
    total_build_time = 0
    global total_prepare_input_time
    total_prepare_input_time = 0
    global total_build_block_time
    total_build_block_time = 0

def get_count():
    global total_sample_core_time
    global total_fetch_prepare_time
    global total_comm_time
    global total_build_time
    global total_prepare_input_time
    global total_build_block_time
    return {
        "total_sample_core_time":total_sample_core_time,
        "total_fetch_prepare_time":total_fetch_prepare_time,
        "total_comm_time":total_comm_time,
        "total_build_time":total_build_time,
        "total_prepare_input_time":total_prepare_input_time,
        "total_build_block_time":total_build_block_time,
    }

def get_node_feature_by_dist(graph:DistributedGraphStore,mailbox:SharedMailBox,query_nid_feature,is_local=True,out_device = torch.device('cuda'),async_op = False):
    ind_dict = None
    if is_local == False:
        ctx = distributed.context._get_default_dist_context()
        if graph.feature_device.type == 'cuda':
            group = ctx.memory_nccl_group
        else:
            group = ctx.memory_gloo_group
    if isinstance(graph.nfeat,DistributedTensor):
        if is_local == False:
            ind_dict = graph.nfeat.all_to_all_ind2ptr(query_nid_feature,group = group)
            node_feat = graph.get_dist_nfeat(**ind_dict,group = group,out_device=out_device,is_async=async_op)
        else:
            node_feat = graph.get_local_nfeat(query_nid_feature,out_device=out_device)
    else:
        node_feat = None
    if mailbox is not None:
        if is_local == False:
            if graph.nfeat is None:
                ind_dict = mailbox.node_memory.all_to_all_ind2ptr(query_nid_feature,group = group)
            mem = mailbox.gather_memory(**ind_dict,group = group,is_async=async_op)
        else:
            mem = mailbox.gather_local_memory(query_nid_feature,compute_device=out_device)
    else:
        mem = None
    return node_feat,mem
def get_edge_feature_by_dist(graph:DistributedGraphStore,query_eid_feature,is_local = True,out_device = torch.device('cuda'),async_op = False):
    if is_local == False:
        ctx = distributed.context._get_default_dist_context()
        if graph.feature_device.type == 'cuda' or graph.use_pin is True:
            group = ctx.memory_nccl_group
        else:
            group = ctx.memory_gloo_group
        return graph.get_dist_efeat(idx = query_eid_feature,group=group,out_device=out_device,is_async=async_op) 
    return graph.get_local_efeat(query_eid_feature,out_device=out_device)

def get_node_all_to_all_route(graph:DistributedGraphStore,mailbox:SharedMailBox,query_nid_feature,out_device = torch.device('cuda')):
    data = []
    ind_dict = None
    ctx = distributed.context._get_default_dist_context()
    if graph.feature_device.type == 'cuda':
        group = ctx.memory_nccl_group
    else:
        group = ctx.memory_gloo_group
    if graph.nfeat is not None:
        ind_dict = graph.nfeat.all_to_all_ind2ptr(query_nid_feature,group = group)
        data.append(graph.get_local_nfeat(ind_dict['recv_ind'],out_device=out_device))
    if mailbox is not None:
        if ind_dict is None:
            ind_dict = graph.nfeat.all_to_all_ind2ptr(query_nid_feature,group = group)
        memory,memory_ts,mail,mail_ts = mailbox.gather_local_memory(ind_dict['recv_ind'],compute_device=out_device)
        memory_ts = memory_ts.reshape(-1,1)
        mail_ts = mail_ts.reshape(mail_ts.shape[0],-1)
        mail = mail.reshape(mail.shape[0],-1)
        data.append(torch.cat((memory,memory_ts,mail,mail_ts),dim = 1))
    if ind_dict is not None:
        return ind_dict,torch.cat(data,dim=1)
    else:
        return None,None
def get_edge_all_to_all_route(graph:DistributedGraphStore,query_eid_feature,out_device = torch.device('cuda')):
    ctx = distributed.context._get_default_dist_context()
    if graph.feature_device.type == 'cuda':
        group = ctx.memory_nccl_group
    else:
        group = ctx.memory_gloo_group
    if graph.efeat is not None:
        ind_dict = graph.nfeat.all_to_all_ind2ptr(query_eid_feature,group = group)
        return ind_dict,graph.get_local_efeat(ind_dict['recv_ind'],out_device=out_device)
    else:
        return None,None
def prepare_input(node_feat, edge_feat, mem_embedding,mfgs,dist_nid=None,dist_eid=None):
    for i,mfg in enumerate(mfgs):
        for b in mfg:
            idx = b.srcdata['__ID']
            if '__ID' in b.edata:
                e_idx = b.edata['__ID']
                b.edata['ID'] = dist_eid[e_idx]
                if edge_feat is not None:
                    b.edata['f'] = edge_feat[e_idx]
            if dist_nid is not None:
                b.srcdata['ID'] = dist_nid[idx]        
            if i == 0:
                if node_feat is not None:
                    b.srcdata['h'] = node_feat[idx]
                if mem_embedding is not None:
                    node_memory,node_memory_ts,mailbox,mailbox_ts = mem_embedding
                    b.srcdata['mem'] = node_memory[idx]
                    b.srcdata['mem_ts'] = node_memory_ts[idx]
                    b.srcdata['mem_input'] = mailbox[idx].reshape(b.srcdata['ID'].shape[0], mailbox.shape[1]*mailbox.shape[2])
                    b.srcdata['mail_ts'] = mailbox_ts[idx].reshape(b.srcdata['ID'].shape[0], mailbox_ts.shape[1])
                    
                    #print(idx.shape[0],b.srcdata['mem_ts'].shape)
    return mfgs

    
def prepare_mailbox(mfgs,mailbox):
    for b in mfgs[0]:
        device = b.device
        local_mask = DistIndex(b.srcdata['ID']).part == dist.get_rank()
        b.srcdata['mem']= torch.zeros(b.srcdata['ID'].shape[0],mailbox.node_memory.shape[1],dtype = mailbox.node_memory.dtype,device = mailbox.node_memory.device)
        b.srcdata['mem_ts'] = torch.zeros(b.srcdata['ID'].shape[0],dtype = mailbox.node_memory_ts.dtype,device = mailbox.node_memory_ts.device)
        b.srcdata['mem_input']= torch.zeros(b.srcdata['ID'].shape[0],mailbox.mailbox_shape, dtype = mailbox.mailbox.dtype,device = mailbox.mailbox.device)
        b.srcdata['mail_ts'] = torch.zeros(b.srcdata['ID'].shape[0],mailbox.mailbox_ts.shape[1],dtype = mailbox.mailbox_ts.dtype,device = mailbox.mailbox_ts.device)
        node_memory,node_memory_ts,node_mailbox,mailbox_ts = mailbox.get_memory(DistIndex(b.srcdata['ID'][local_mask]).loc,local=True)
        b.srcdata['mem'][local_mask] = node_memory.to(device)
        b.srcdata['mem_ts'][local_mask] = node_memory_ts.to(device)
        b.srcdata['mem_input'][local_mask] = node_mailbox.reshape(local_mask.sum(), -1).to(device)
        b.srcdata['mail_ts'][local_mask] = mailbox_ts.to(device)

import os
def to_block(graph,data, sample_out,device = torch.device('cuda'),unique = True):
    if len(sample_out) > 1:
        sample_out,metadata = sample_out
    else:
        metadata = None
    #to_block(metadata['src_pos_index'],metadata['dst_pos_index'],metadata['dst_neg_index'],
    #         metadata['seed'],metadata['seed_ts'],graph.nids_mapper,graph.eids_mapper,#device.type if "cpu" else str(device.index))
    #root_len = len(metadata.pop('seed'))
    #eid_inv = metadata.pop('eid_inv').clone()
    #print('data {} {}\n'.format(data.edges,data.ts))
    #first_block_id = metadata.pop('first_block_id').clone()
    #print('first_block_id {}\n'.format(first_block_id))
    #block_node_list = metadata.pop('block_node_list').clone()
    #print('block_node_list {}\n'.format(block_node_list))
    #unq_id = metadata.pop('unq_id').clone()
    #print('unq id {}'.format(unq_id))
    #dist_nid = metadata.pop('dist_nid').clone().to(device) 
    #dist_eid = metadata.pop('dist_eid').clone().to(device) 
    #print('dist nid {} dist eid {}\n'.format(dist_nid,dist_eid))
    #print('block node list edge {} {}'.format(
    #                                                      graph.ids[DistIndex(dist_nid[block_node_list[0,#unq_id]]).loc.to('cpu')],block_node_list[1,unq_id]
    eid_len = [ret.eid().shape[0] for ret in sample_out ]
    # print(sample_out)
    
    t0 = time.time()
    eid = [ret.eid() for ret in sample_out]
    dst = [ret.sample_nodes() for ret in sample_out]
    dst_ts = [ret.sample_nodes_ts() for ret in sample_out]
    dst = torch.cat(dst,dim = 0)
    dst_ts = torch.cat(dst_ts,dim = 0).to(device)
    eid_len = [e.shape[0] for e in eid ]
    eid_mapper: torch.Tensor = graph.eids_mapper
    nid_mapper: torch.Tensor = graph.nids_mapper
    eid_tensor = torch.cat(eid,dim = 0).to(eid_mapper.device)
    dist_eid = eid_mapper[eid_tensor].to(device)
    dist_eid,eid_inv = dist_eid.unique(return_inverse=True)
    src_node = dst.to(graph.nids_mapper.device)
    #print(src_node)
    src_ts = None  
    if metadata is None:
        root_node = data.nodes.to(graph.nids_mapper.device)
        root_len = [root_node.shape[0]]
        if hasattr(data,'ts'):
            src_ts = torch.cat([data.ts,
                                graph.sample_graph['ts'][eid_tensor].to(device)])
    elif 'seed' in metadata:
        root_node = metadata.pop('seed').to(graph.nids_mapper.device)
        root_len = root_node.shape[0]
        if 'seed_ts' in metadata:
            src_ts = torch.cat([metadata.pop('seed_ts').to(device),\
                                dst_ts.to(device)])
        for k in metadata:
            if isinstance(metadata[k],torch.Tensor):
                metadata[k] = metadata[k].to(device)
    nid_tensor = torch.cat([root_node,src_node],dim = 0)
    dist_nid = nid_mapper[nid_tensor].to(device)
    #print(CountComm.origin_local,CountComm.origin_remote)
    #for p in range(dist.get_world_size()):
    #    print((DistIndex(dist_nid).part == p).sum().item())
    #CountComm.origin_local = (DistIndex(dist_nid).part == dist.get_rank()).sum().item()
    #CountComm.origin_remote =(DistIndex(dist_nid).part != dist.get_rank()).sum().item()
    dist_nid,nid_inv = dist_nid.unique(return_inverse=True)
    #print('nid_tensor {} \n nid {}\n'.format(nid_tensor,dist_nid))

    """
    对于同id和同时间的节点去重取得index
    """
    if unique:
        block_node_list,unq_id = torch.stack((nid_inv.to(torch.float64),src_ts.to(torch.float64))).unique(dim = 1,return_inverse=True)
        #print(block_node_list.shape,unq_id.shape)
        first_index,_ = torch_scatter.scatter_min(torch.arange(unq_id.shape[0],device=unq_id.device,dtype=unq_id.dtype),unq_id)
        first_mask = torch.zeros(unq_id.shape[0],device = unq_id.device,dtype=torch.bool)
        first_mask[first_index] = True
        first_index = unq_id[first_mask]
        first_block_id = torch.empty(first_index.shape[0],device=unq_id.device,dtype=unq_id.dtype)
        first_block_id[first_index] = torch.arange(first_index.shape[0],device=first_index.device,dtype=first_index.dtype)
        first_block_id = first_block_id[unq_id].contiguous()
        block_node_list = block_node_list[:,first_index]
        #print('first block id {}\n unq id {} \n block_node_list {}\n'.format(first_block_id,unq_id,block_node_list))
        for k in metadata:
            if isinstance(metadata[k],torch.Tensor):
                #print('{}:{}\n'.format(k,metadata[k]))
                metadata[k] = first_block_id[metadata[k]]
                #print('{}:{}\n'.format(k,metadata[k]))
        t2 = time.time()
    
        def build_block():
            mfgs = list()
            col_len = 0
            row_len = root_len
            col = first_block_id[:row_len]
            max_row = col.max().item()+1 if col.numel() > 0 else 0
            for r in range(len(eid_len)):
                elen = eid_len[r]
                row = first_block_id[row_len:row_len+elen]
                col0 = col[sample_out[r].src_index().to(device)]
                b = dgl.create_block((row,col0),
                                     num_src_nodes = max(max_row, row.max().item()+1 if row.numel() > 0 else 0),
                                     num_dst_nodes = max_row,
                                     device = device)

                max_row = max(max_row, row.max().item()+1 if row.numel() > 0 else 0)
                idx = block_node_list[0,b.srcnodes()].to(torch.long)
                e_idx = eid_inv[col_len:col_len+elen]
                b.srcdata['__ID'] = idx
                
                if sample_out[r].delta_ts().shape[0] > 0:
                    b.edata['dt'] = sample_out[r].delta_ts().to(device)
                b.srcdata['ts'] = block_node_list[1,b.srcnodes()].to(torch.float)
                #weight =  sample_out[r].sample_weight()
                #if(weight.shape[0] > 0):
                #    b.edata['weight'] = 1/torch.clamp(sample_out[r].sample_weight(),0.0001).to(b.device)
                b.edata['__ID'] = e_idx
                col = row
                col_len += eid_len[r]
                row_len += eid_len[r]
                mfgs.append(b)
            mfgs = list(map(list, zip(*[iter(mfgs)])))
            mfgs.reverse()
            return data,mfgs,metadata
        data,mfgs,metadata = build_block()
        return (data,mfgs,metadata),dist_nid,dist_eid
    
    
    else:
        def build_block():
            mfgs = list()
            col = torch.arange(0,root_len,device = device)
            col_len = 0
            row_len = root_len
            for r in range(len(eid_len)):
                elen = eid_len[r]
                row = torch.arange(row_len,row_len+elen,device = device)
                b = dgl.create_block((row,col[sample_out[r].src_index().to(device)]),
                                     num_src_nodes = row_len + elen,
                                     num_dst_nodes = row_len,
                                     device = device)
                idx = nid_inv[0:row_len + elen]
                e_idx = eid_inv[col_len:col_len+elen]
                b.srcdata['__ID'] = idx
                if sample_out[r].delta_ts().shape[0] > 0:
                    b.edata['dt'] = sample_out[r].delta_ts().to(device)
                if src_ts is not None:
                    b.srcdata['ts'] = src_ts[0:row_len + eid_len[r]]
                b.edata['__ID'] = e_idx
                col = row
                col_len += eid_len[r]
                row_len += eid_len[r]

                mfgs.append(b)
            mfgs = list(map(list, zip(*[iter(mfgs)])))
            mfgs.reverse()
            return data,mfgs,metadata

        data,mfgs,metadata = build_block()
        return (data,mfgs,metadata),dist_nid,dist_eid

def to_reversed_block(graph,data, sample_out,device = torch.device('cuda'),unique = True,identity=False):
    if len(sample_out) > 1:
        sample_out,metadata = sample_out
    else:
        metadata = None
    nid_mapper: torch.Tensor = graph.nids_mapper
    #print('reverse block {}\n'.format(identity))
    if identity is False:
        assert len(sample_out) == 1
        ret = sample_out[0]
        eid_len = ret.eid().shape[0]
        t0 = time.time()
        dst_ts = ret.sample_nodes_ts().to(device)
        dst = nid_mapper[ret.sample_nodes()].to(device)
        dist_eid = torch.tensor([],dtype=torch.long,device=device)
        src_index = ret.src_index().to(device)
    else:
        #print('is jodie')
        #print(sample_out)
        src_index = torch.tensor([],dtype=torch.long,device=device)
        dst = torch.tensor([],dtype=torch.long,device=device)
        dist_eid = torch.tensor([],dtype=torch.long,device=device)
    if metadata is None:
        root_node = data.nodes.to(graph.nids_mapper.device)
        root_len = [root_node.shape[0]]
        root_ts = data.ts.to(device)
                                
    elif 'seed' in metadata:
        root_node = metadata.pop('seed').to(graph.nids_mapper.device)
        root_len = root_node.shape[0]
        if 'seed_ts' in metadata:
            root_ts = metadata.pop('seed_ts').to(device)
        for k in metadata:
            if isinstance(metadata[k],torch.Tensor):
                metadata[k] = metadata[k].to(device)
    src_node = root_node
    src_ts = root_ts
    nid_tensor = torch.cat([root_node],dim = 0)
    dist_nid = nid_mapper[nid_tensor].to(device)
    CountComm.origin_local = (DistIndex(dist_nid).part == dist.get_rank()).sum().item()
    CountComm.origin_remote =(DistIndex(dist_nid).part != dist.get_rank()).sum().item()
    dist_nid,nid_inv = dist_nid.unique(return_inverse=True)

    """
    对于同id和同时间的节点去重取得index
    """
    block_node_list,unq_id = torch.stack((nid_inv.to(torch.float64),src_ts.to(torch.float64))).unique(dim = 1,return_inverse=True)
    first_index,_ = torch_scatter.scatter_min(torch.arange(unq_id.shape[0],device=unq_id.device,dtype=unq_id.dtype),unq_id)
    first_mask = torch.zeros(unq_id.shape[0],device = unq_id.device,dtype=torch.bool)
    first_mask[first_index] = True
    first_index = unq_id[first_mask]
    first_block_id = torch.empty(first_index.shape[0],device=unq_id.device,dtype=unq_id.dtype)
    first_block_id[first_index] = torch.arange(first_index.shape[0],device=first_index.device,dtype=first_index.dtype)
    first_block_id = first_block_id[unq_id].contiguous()
    block_node_list = block_node_list[:,first_index]
    for k in metadata:
        if isinstance(metadata[k],torch.Tensor):
            metadata[k] = first_block_id[metadata[k]]
    t2 = time.time()
    
    def build_block():
        mfgs = list()
        col_len = 0
        row_len = root_len
        col = first_block_id[:row_len]
        max_row = col.max().item()+1
        #print(src_index,dst)
        b = dgl.create_block((col[src_index].to(device),
                                torch.arange(dst.shape[0],device=device,dtype=torch.long)),num_src_nodes=first_block_id.max().item()+1,
                                num_dst_nodes=dst.shape[0])
        idx = block_node_list[0,b.srcnodes()].to(torch.long)
        b.srcdata['__ID'] = idx
        b.srcdata['ts'] = block_node_list[1,b.srcnodes()].to(torch.float)
        b.dstdata['ID'] = dst
        mfgs.append(b)
        mfgs = list(map(list, zip(*[iter(mfgs)])))
        mfgs.reverse()
        return data,mfgs,metadata
    data,mfgs,metadata = build_block()
    return (data,mfgs,metadata),dist_nid,dist_eid



import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) 

def graph_sample(graph,sampler,sample_fn,data,neg_sampling = None,out_device = torch.device('cuda'),nid_mapper = None,eid_mapper=None,reversed=False):
    t_s  = time.time()
    param = {'is_unique':False,'nid_mapper':nid_mapper,'eid_mapper':eid_mapper,'out_device':out_device}
    out = sample_fn(sampler,data,neg_sampling,**param)
    #print(sampler.policy)
    if reversed is False:
        out,dist_nid,dist_eid = to_block(graph,data,out,out_device)
    else:
        out,dist_nid,dist_eid = to_reversed_block(graph,data,out,out_device,identity=(sampler.policy=='identity'))
    t_e = time.time()
    #print(t_e-t_s)
    return out,dist_nid,dist_eid

def prepare_for_pin(dist_nid,dist_eid):
    pass                                                                                                                                                                                                                                       
"""
def graph_sample(graph, sampler:BaseSampler,
                      sample_fn, data, 
                      neg_sampling = None,
                      mailbox = None,
                      device = torch.device('cuda'),
                      async_op = False,
                      local_embedding = False,
                      is_train = True
                      ):
    t_s = tt.start()
    if async_op == False:
        t0 = time.time()
        if sample_fn != sample_from_local_temporal_edges:
            out = sample_fn(sampler,data,neg_sampling)
        else:
            out = sample_fn(sampler,data,graph.nids_mapper,neg_sampling)
        t1 = time.time()
        #print('sample time is {}'.format(t1-t0))
        #torch.cuda.synchronize()
        tt.time_sample+=tt.elapsed(t_s)
        t_s = tt.start()
        if sampler.historical == False:
            #b = build_unq_block(graph,data,out,mailbox,device,use_cache=False,use_dist=True)
            b = to_block(graph,data,out,mailbox,device,is_train=is_train)
            #b = to_tgl_sample_block(out,graph)
        else:
            #b = build_unq_block(graph,data,out,mailbox,device,use_cache=True,use_dist=False)
            b = historical_to_block(graph,data,out,mailbox,torch.cuda.current_device())
        #torch.cuda.synchronize()
        tt.pre_batch += tt.elapsed(t_s)
        return b
    else:
        global executor
        if sample_fn != sample_from_local_temporal_edges:
            fut = executor.submit(sample_fn,sampler = sampler,data = data,neg_sampling = neg_sampling)
        else:
            fut = executor.submit(sample_fn,sampler = sampler,data = data,id_mapper = graph.nids_mapper,neg_sampling = neg_sampling)
        return fut
def async_to_block(graph,data,sampler,sample_out,mailbox,device):
    out = sample_out
    if sampler.historical == False:
        return to_block(graph,data,out,mailbox,device)
    else:
        return historical_to_block(graph,data,out,mailbox,torch.cuda.current_device())
"""
def sample_from_nodes(sampler:BaseSampler,  data:DataSet, **kwargs):
    out = sampler.sample_from_nodes(nodes=data.nodes.reshape(-1))
    #out.metadata = None
    return out

def sample_from_edges(sampler:BaseSampler,  
                      data:DataSet, 
                      neg_sampling:NegativeSampling = None):
    out = sampler.sample_from_edges(edges = data.edges.to('cpu'), 
                                    neg_sampling=neg_sampling)
    return out
    

def sample_from_temporal_nodes(sampler:BaseSampler,data:DataSet,
                               **kwargs):
    out = sampler.sample_from_nodes(nodes=data.nodes.reshape(-1),
                                    ts = data.ts.reshape(-1),
                                    **kwargs)
    #out.metadata = None
    return out


def sample_from_temporal_edges(sampler:BaseSampler, data:DataSet,
                               neg_sampling: NegativeSampling = None,
                               **kwargs):
    out = sampler.sample_from_edges(edges=data.edges.to('cpu'),
                                    ets=data.ts.to('cpu'),
                                    neg_sampling = neg_sampling,
                                    **kwargs
                                    )
    return out

def sample_from_local_temporal_edges(sampler:BaseSampler, data:DataSet,id_mapper,
                               neg_sampling: NegativeSampling = None):
    out = sampler.sample_from_edges_with_distributed_dst(
                                    data,id_mapper,
                                    neg_sampling = neg_sampling
                                    )
    return out

class SAMPLE_TYPE:
    SAMPLE_FROM_NODES = sample_from_nodes
    SAMPLE_FROM_EDGES = sample_from_edges
    SAMPLE_FROM_TEMPORAL_NODES = sample_from_temporal_nodes
    SAMPLE_FROM_TEMPORAL_EDGES = sample_from_temporal_edges
    SAMPLE_FROM_LOCAL_TEMPORAL_EDGES = sample_from_local_temporal_edges
    