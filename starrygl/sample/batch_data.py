import os
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
from starrygl.sample import dual_dedup_view
from starrygl.sample import direct_segment_block
from starrygl.module import triton_edge_projection
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
    dual_dedup_view.reset()

def get_count():
    global total_sample_core_time
    global total_fetch_prepare_time
    global total_comm_time
    global total_build_time
    global total_prepare_input_time
    global total_build_block_time
    dual_stats = dual_dedup_view.stats()
    return {
        "total_sample_core_time":total_sample_core_time,
        "total_fetch_prepare_time":total_fetch_prepare_time,
        "total_comm_time":total_comm_time,
        "total_build_time":total_build_time,
        "total_prepare_input_time":total_prepare_input_time,
        "total_build_block_time":total_build_block_time,
        "dual_dedup_view": dual_stats,
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
    compact_gru = os.environ.get('MEMSHARE_COMPACT_GRU_UPDATE') == 'on'
    compact_memory_ops_enabled = (
        os.environ.get('MEMSHARE_COMPACT_MEMORY_OPS') == 'on')
    for i,mfg in enumerate(mfgs):
        for b in mfg:
            idx = b.srcdata['__ID']
            compact_memory_ops = (
                compact_memory_ops_enabled
                and mem_embedding is not None
                and getattr(b, '_memshare_hierarchical_compaction', False)
            )
            if dual_dedup_view.enabled():
                dual_dedup_view.attach(b, node_feat, edge_feat, mem_embedding)
            if '__ID' in b.edata:
                e_idx = b.edata['__ID']
                b.edata['ID'] = dist_eid[e_idx]
                if edge_feat is not None:
                    compact_kv = os.environ.get('MEMSHARE_COMPACT_KV_PROJECTION') == 'on'
                    if triton_edge_projection.enabled() or compact_kv:
                        b.edata['_edge_to_eid'] = e_idx
                        b._memshare_edge_feat_object = edge_feat
                    if not triton_edge_projection.enabled() or compact_kv:
                        b.edata['f'] = edge_feat[e_idx]
            if dist_nid is not None:
                b.srcdata['ID'] = dist_nid[idx]        
            if i == 0:
                if node_feat is not None:
                    if compact_memory_ops:
                        b._memshare_node_feat_object = node_feat
                    else:
                        b.srcdata['h'] = node_feat[idx]
                if mem_embedding is not None:
                    node_memory,node_memory_ts,mailbox,mailbox_ts = mem_embedding
                    b.srcdata['mem'] = node_memory[idx]
                    b.srcdata['mem_ts'] = node_memory_ts[idx]
                    if compact_gru or compact_memory_ops_enabled:
                        b._memshare_memory_object = node_memory
                        b._memshare_mailbox_object = mailbox.reshape(
                            mailbox.shape[0], -1)
                        b._memshare_mailbox_ts_object = mailbox_ts.reshape(
                            mailbox_ts.shape[0], -1)
                    if compact_memory_ops:
                        b._memshare_dist_nid_object = dist_nid
                    if not compact_gru:
                        b.srcdata['mem_input'] = mailbox[idx].reshape(b.srcdata['ID'].shape[0], -1)
                    b.srcdata['mail_ts'] = mailbox_ts[idx].reshape(b.srcdata['ID'].shape[0], -1)
                    
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


def to_block(graph,data, sample_out,device = torch.device('cuda'),unique = True,
             state_unique = None):
    if state_unique is None:
        state_unique = unique
    if len(sample_out) > 1:
        sample_out,metadata = sample_out
    else:
        metadata = None
    if metadata is not None and '_hierarchical_compaction' in metadata:
        compact = metadata.pop('_hierarchical_compaction')
        (dist_nid, dist_eid, state_to_uid, state_ts, root_to_state,
         row, col, edge_to_eid, delta_ts, sizes) = compact
        num_src, num_dst = sizes.tolist()
        (dist_nid, dist_eid, state_to_uid, state_ts, root_to_state,
         row, col, edge_to_eid, delta_ts) = (
            tensor.to(device) for tensor in compact[:-1]
        )
        if os.environ.get('MEMSHARE_FUSED_SEGMENT_ATTENTION') == 'on':
            block = direct_segment_block.create_block(
                row, col, num_src, num_dst)
        else:
            block = dgl.create_block(
                (row, col), num_src_nodes=num_src,
                num_dst_nodes=num_dst, device=device)
        block.srcdata['__ID'] = state_to_uid
        block.srcdata['ts'] = state_ts.float()
        block.edata['__ID'] = edge_to_eid
        block.edata['dt'] = delta_ts
        block._memshare_hierarchical_compaction = True

        seed = metadata.pop('seed')
        seed_ts = metadata.pop('seed_ts')
        if seed.numel() != root_to_state.numel() \
                or seed_ts.numel() != root_to_state.numel():
            raise RuntimeError('hierarchical root mapping does not match seeds')
        root_index_keys = (
            'src_pos_index', 'dst_pos_index',
            'src_neg_index', 'dst_neg_index',
        )
        for key, value in metadata.items():
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device)
            metadata[key] = (
                root_to_state[value] if key in root_index_keys else value
            )
        return (data, [[block]], metadata), dist_nid, dist_eid
    eid_len = [ret.eid().shape[0] for ret in sample_out ]
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
    if unique:
        dist_eid,eid_inv = dist_eid.unique(return_inverse=True)
    else:
        dist_eid, order = dist_eid.sort()
        eid_inv = torch.empty_like(order)
        eid_inv[order] = torch.arange(order.numel(), device=device)
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
    if unique:
        dist_nid,nid_inv = dist_nid.unique(return_inverse=True)
    else:
        dist_nid, order = dist_nid.sort()
        nid_inv = torch.empty_like(order)
        nid_inv[order] = torch.arange(order.numel(), device=device)

    """
    对于同id和同时间的节点去重取得index
    """
    if state_unique:
        block_node_list,unq_id = torch.stack((nid_inv.to(torch.float64),src_ts.to(torch.float64))).unique(dim = 1,return_inverse=True)
        #print(block_node_list.shape,unq_id.shape)
        first_index,_ = torch_scatter.scatter_min(torch.arange(unq_id.shape[0],device=unq_id.device,dtype=unq_id.dtype),unq_id)
        first_mask = torch.zeros(unq_id.shape[0],device = unq_id.device,dtype=torch.bool)
        first_mask[first_index] = True
        first_index = unq_id[first_mask]
        first_block_id = torch.empty(first_index.shape[0],device=unq_id.device,dtype=unq_id.dtype)
        first_block_id[first_index] = torch.arange(first_index.shape[0],device=first_index.device,dtype=first_index.dtype)
        first_block_id = first_block_id[unq_id].contiguous()
        block_node_list = block_node_list[:,first_index].contiguous()
        #print('first block id {}\n unq id {} \n block_node_list {}\n'.format(first_block_id,unq_id,block_node_list))
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
            for r in range(len(eid_len)):
                elen = eid_len[r]
                row = first_block_id[row_len:row_len+elen]
                col0 = col[sample_out[r].src_index().to(device)]
                row_max = row.max().item()+1 if row.numel() > 0 else 0
                num_src_nodes = max(max_row, row_max)
                if os.environ.get('MEMSHARE_FUSED_SEGMENT_ATTENTION') == 'on':
                    b = direct_segment_block.create_block(
                        row, col0, num_src_nodes, max_row)
                else:
                    b = dgl.create_block(
                        (row, col0), num_src_nodes=num_src_nodes,
                        num_dst_nodes=max_row, device=device)

                max_row = num_src_nodes
                idx = block_node_list[0,b.srcnodes()].to(torch.long)
                e_idx = eid_inv[col_len:col_len+elen]
                b.srcdata['__ID'] = idx
                
                if sample_out[r].delta_ts().shape[0] > 0:
                    b.edata['dt'] = sample_out[r].delta_ts().to(device)
                b.srcdata['ts'] = block_node_list[1,b.srcnodes()].to(torch.float)
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

def to_reversed_block(graph,data, sample_out,device = torch.device('cuda'),
                      unique = True,identity=False,state_unique = None):
    if state_unique is None:
        state_unique = unique
    if len(sample_out) > 1:
        sample_out,metadata = sample_out
    else:
        metadata = None
    nid_mapper: torch.Tensor = graph.nids_mapper
    #print('reverse block {}\n'.format(identity))
    if identity is False:
        assert len(sample_out) == 1
        ret = sample_out[0]
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
    if unique:
        dist_nid,nid_inv = dist_nid.unique(return_inverse=True)
    else:
        dist_nid, order = dist_nid.sort()
        nid_inv = torch.empty_like(order)
        nid_inv[order] = torch.arange(order.numel(), device=device)

    """
    对于同id和同时间的节点去重取得index
    """
    if state_unique:
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
    else:
        first_block_id = torch.arange(nid_inv.shape[0], device=device)
        block_node_list = torch.stack((nid_inv, src_ts))
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
    if reversed and getattr(sampler, 'hierarchical_compaction', False):
        raise RuntimeError('hierarchical compaction does not support reversed blocks')
    param = {'is_unique':False,'nid_mapper':nid_mapper,'eid_mapper':eid_mapper,'out_device':out_device}
    out = sample_fn(sampler,data,neg_sampling,**param)
    #print(sampler.policy)
    object_unique = os.environ.get('MEMSHARE_BLOCK_DEDUP', 'on') == 'on'
    state_setting = os.environ.get('MEMSHARE_STATE_DEDUP', 'inherit')
    state_unique = object_unique if state_setting == 'inherit' else state_setting == 'on'
    if reversed is False:
        out,dist_nid,dist_eid = to_block(
            graph, data, out, out_device,
            unique=object_unique,state_unique=state_unique)
    else:
        out,dist_nid,dist_eid = to_reversed_block(
            graph, data, out, out_device,
            unique=object_unique,state_unique=state_unique,
            identity=(sampler.policy=='identity'))
    t_e = time.time()
    #print(t_e-t_s)
    return out,dist_nid,dist_eid

def prepare_for_pin(dist_nid,dist_eid):
    pass                                                                                                                                                                                                                                       
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
