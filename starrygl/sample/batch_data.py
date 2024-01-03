from typing import List, Tuple
import torch
import torch.distributed as dist
from starrygl.distributed.utils import DistributedTensor
from starrygl.module.memorys import MailBox
from starrygl.sample.cache.fetch_cache import FetchFeatureCache
from starrygl.sample.graph_core import DataSet
from starrygl.sample.graph_core import DistributedGraphStore
from starrygl.sample.sample_core.base import BaseSampler, NegativeSampling
import dgl

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
def prepare_input(node_feat, edge_feat, mem_embedding,mfgs,dist_nid,dist_eid):
    for mfg in mfgs:
        for i,b in enumerate(mfg):
            e_idx = b.edata['ID']
            idx = b.srcdata['ID']
            b.edata['ID'] = dist_eid[e_idx]
            b.srcdata['ID'] = dist_nid[idx]
            if edge_feat is not None:
                b.edata['f'] = edge_feat[e_idx]
            if i == 0:
                if node_feat is not None:
                    b.srcdata['h'] = node_feat[idx]
                if mem_embedding is not None:
                    node_memory,node_memory_ts,mailbox,mailbox_ts = mem_embedding
                    b.srcdata['mem'] = node_memory[idx]
                    b.srcdata['mem_ts'] = node_memory_ts[idx]
                    b.srcdata['mem_input'] = mailbox[idx].reshape(b.srcdata['ID'].shape[0], -1)
                    b.srcdata['mail_ts'] = mailbox_ts[idx]
                    #print(idx.shape[0],b.srcdata['mem_ts'].shape)
        return mfgs

def to_block(graph: DistributedGraphStore, data, sample_out, mailbox:MailBox = None,device = torch.device('cuda'),group = None):
    if len(sample_out) > 1:
        sample_out,metadata = sample_out
    else:
        metadata = None
    eid = [ret.eid() for ret in sample_out]
    eid_len = [e.shape[0] for e in eid ]
    eid_mapper: torch.Tensor = graph.eids_mapper
    nid_mapper: torch.Tensor = graph.nids_mapper
    eid_tensor = torch.cat(eid,dim = 0).to(eid_mapper.device)
    dist_eid = eid_mapper[eid_tensor].to(device)
    dist_eid,eid_inv = dist_eid.unique(return_inverse=True)
    src_node = graph.sample_graph['edge_index'][0,eid_tensor*2].to(graph.nids_mapper.device)
    src_ts = None  
    if metadata is None:
        root_node = data.nodes.to(graph.nids_mapper.device)
        root_len = [root_node.shape[0]]
        if hasattr(data,'ts'):
            src_ts = torch.cat([data.ts,
                                graph.sample_graph['ts'][eid_tensor*2].to(device)])
    elif 'seed' in metadata:
        root_node = metadata.pop('seed').to(graph.nids_mapper.device)
        root_len = root_node.shape[0]
        if 'seed_ts' in metadata:
            src_ts = torch.cat([metadata.pop('seed_ts').to(device),\
                                graph.sample_graph['ts'][eid_tensor*2].to(device)])
        for k in metadata:
            metadata[k] = metadata[k].to(device)

    nid_tensor = torch.cat([root_node,src_node],dim = 0)
    dist_nid = nid_mapper[nid_tensor].to(device)
    dist_nid,nid_inv = dist_nid.unique(return_inverse = True)
    
    fetchCache = FetchFeatureCache.getFetchCache()
    if fetchCache is None:
        if isinstance(graph.edge_attr,DistributedTensor):
            ind_dict = graph.edge_attr.all_to_all_ind2ptr(dist_eid,group = group)
            edge_feat = graph.edge_attr.all_to_all_get(group = group,**ind_dict)
        else:
            edge_feat = graph._get_edge_attr(dist_eid)
        ind_dict = None
        if isinstance(graph.x,DistributedTensor):
            ind_dict = graph.x.all_to_all_ind2ptr(dist_nid,group = group)
            node_feat = graph.x.all_to_all_get(group = group,**ind_dict)
        else:
            node_feat = graph._get_node_attr(dist_nid)
        if mailbox is not None:
            if torch.distributed.get_world_size() > 1:
                if node_feat is None:
                    ind_dict = mailbox.node_memory.all_to_all_ind2ptr(dist_nid,group = group)
                mem = mailbox.gather_memory(**ind_dict)
            else:
                mem = mailbox.get_memory(dist_nid)
        else:
            mem = None
    else:
        raw_nid = torch.empty_like(dist_nid)
        raw_eid = torch.empty_like(dist_eid)
        nid_tensor = nid_tensor.to(device)
        eid_tensor = eid_tensor.to(device)
        raw_nid[nid_inv] = nid_tensor
        raw_eid[eid_inv] = eid_tensor
        node_feat,edge_feat,mem = fetchCache.fetch_feature(raw_nid,
                                 dist_nid,raw_eid,
                                 dist_eid)
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
            b.srcdata['ID'] = idx
            if sample_out[r].delta_ts().shape[0] > 0:
                b.edata['dt'] = sample_out[r].delta_ts().to(device)
            if src_ts is not None:
                b.srcdata['ts'] = src_ts[0:row_len + eid_len[r]]
            b.edata['ID'] = e_idx
            col = row
            col_len += eid_len[r]
            row_len += eid_len[r]
            mfgs.append(b)
        mfgs = list(map(list, zip(*[iter(mfgs)])))
        mfgs.reverse()
        return data,mfgs,metadata
    data,mfgs,metadata = build_block()
    mfgs = prepare_input(node_feat,edge_feat,mem,mfgs,dist_nid,dist_eid)
        #return build_block(node_feat,edge_feat,mem)#data,mfgs,metadata
    return (data,mfgs,metadata)


def graph_sample(graph, sampler:BaseSampler,
                      sample_fn, data, 
                      neg_sampling = None,
                      mailbox = None,
                      device = torch.device('cuda'),
                      async_op = False):
    out = sample_fn(sampler,data,neg_sampling)
    if async_op == False:
        return to_block(graph,data,out,mailbox,device)
    else:
        manger =  getPipelineManger()
        future = manger.submit('lookup',to_block,{'graph':graph,'data':data,\
                                                  'sample_out':out,\
                                                  'mailbox':mailbox,\
                                                  'device':device})
        return future

def sample_from_nodes(sampler:BaseSampler,  data:DataSet, **kwargs):
    out = sampler.sample_from_nodes(nodes=data.nodes.reshape(-1))
    #out.metadata = None
    return out

def sample_from_edges(sampler:BaseSampler,  
                      data:DataSet, 
                      neg_sampling:NegativeSampling = None):
    edge_label = data.labels if hasattr(data,'labels') else None
    out = sampler.sample_from_edges(edges = data.edges, 
                                    neg_sampling=neg_sampling)
    return out
    

def sample_from_temporal_nodes(sampler:BaseSampler,data:DataSet,
                               **kwargs):
    out = sampler.sample_from_nodes(nodes=data.nodes.reshape(-1),
                                    ts = data.ts.reshape(-1))
    #out.metadata = None
    return out


def sample_from_temporal_edges(sampler:BaseSampler, data:DataSet,
                               neg_sampling: NegativeSampling = None):
    edge_label = data.labels if hasattr(data,'labels') else None
    out = sampler.sample_from_edges(edges=data.edges.to('cpu'),
                                    ets=data.ts.to('cpu'),
                                    neg_sampling = neg_sampling
                                    )
    return out

class SAMPLE_TYPE:
    SAMPLE_FROM_NODES = sample_from_nodes,
    SAMPLE_FROM_EDGES = sample_from_edges,
    SAMPLE_FROM_TEMPORAL_NODES = sample_from_temporal_nodes,
    SAMPLE_FROM_TEMPORAL_EDGES = sample_from_temporal_edges
    