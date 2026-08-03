from typing import List
import torch.distributed as dist
import torch
import torch_scatter
from starrygl.distributed.context import _get_default_dist_context
import pandas as pd
import numpy as np
import os

from starrygl.distributed.utils import DistIndex
from starrygl.module.historical_cache import CachePushRoute, HistoricalCache
from starrygl.sample.graph_core import DistributedGraphStore
from starrygl.sample.part_utils.partition_tgnn import partition_load

def load_feat(d, node_num = 0, edge_num = 0, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('/mnt/nfs/fzz/TGL-DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('/mnt/nfs/fzz/TGL-DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    if node_feats is None:
        node_feats = torch.randn(node_num,rand_dn)
    else:
        if node_feats.shape[1] < rand_dn:
            _node_feats = torch.zeros(node_num,rand_dn)
            _node_feats[:,:node_feats.shape[1]] = node_feats
            node_feats = _node_feats
    #else:
        #if d == 'WIKI':
        #    node_feats = torch.randn(9228, 16).to(torch.float32)
    edge_feats = None
    if os.path.exists('/mnt/nfs/fzz/TGL-DATA//{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('/mnt/nfs/fzz/TGL-DATA//{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if edge_feats is None :
        edge_feats = torch.randn(edge_num, rand_de)
    else:
        if edge_feats.shape[1] < rand_de:
            _edge_feats = torch.zeros(edge_num,rand_de)
            _edge_feats[:,:edge_feats.shape[1]] = edge_feats
            edge_feats = _edge_feats
    edge_feats = torch.cat([edge_feats, torch.zeros((1, edge_feats.shape[1]), dtype=edge_feats.dtype)])
    #if d == 'GDELT':
    #    node_feats = node_feats.to(torch.float16)
    #    edge_feats = edge_feats.to(torch.float16)
    return node_feats,edge_feats

def load_graph(d):
    df = pd.read_csv('/mnt/nfs/fzz/TGL-DATA/{}/edges.csv'.format(d))
    
    return df


def load_edge_tensors(d):
    """Load immutable edge columns, preferring an existing tensor cache."""

    cache_dir = '/mnt/nfs/fzz/TGL-DATA/{}/edges_cache'.format(d)
    cache_files = {
        'src': os.path.join(cache_dir, 'src.npy'),
        'dst': os.path.join(cache_dir, 'dst.npy'),
        'time': os.path.join(cache_dir, 'time.npy'),
        'ext_roll': os.path.join(cache_dir, 'ext_roll.npy'),
    }
    if all(os.path.exists(path) for path in cache_files.values()):
        src = torch.from_numpy(np.load(cache_files['src'], mmap_mode='c')).long()
        dst = torch.from_numpy(np.load(cache_files['dst'], mmap_mode='c')).long()
        ts = torch.from_numpy(np.load(cache_files['time'], mmap_mode='c')).long()
        ext_roll = torch.from_numpy(
            np.load(cache_files['ext_roll'], mmap_mode='c')
        ).long()
        return src, dst, ts, ext_roll

    cache_path = '/mnt/nfs/fzz/TGL-DATA/{}/edges_cache.pt'.format(d)
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location='cpu')
        return (
            obj['src'].long(),
            obj['dst'].long(),
            obj['time'].long(),
            obj['ext_roll'].long(),
        )

    df = load_graph(d)
    src = torch.from_numpy(np.array(df.src.values)).long()
    dst = torch.from_numpy(np.array(df.dst.values)).long()
    ts = torch.from_numpy(np.array(df.time.values)).long()
    ext_roll = torch.from_numpy(np.array(df.ext_roll.values)).long()
    return src, dst, ts, ext_roll

def build_shared_index(ids,shared_ids,to_shared_idexs):
    ctx = _get_default_dist_context()
    rank = ctx.memory_group_rank
    world_size = ctx.memory_group_size
    ikw = dict(dtype=torch.long, device=torch.device('cpu'))
    num_nodes = torch.zeros(1, **ikw)
    num_nodes[0] = ids.max().item()+1
    dist.all_reduce(num_nodes, op=dist.ReduceOp.MAX,group=ctx.gloo_group)
    all_ids: List[torch.Tensor] = [None] * world_size
    shared_id_index:  List[torch.Tensor] = [None] * world_size
    dist.all_gather_object(all_ids,ids,group=ctx.memory_gloo_group)
    dist.all_gather_object(shared_id_index,to_shared_idexs,group=ctx.memory_gloo_group)
    part_mp = torch.empty(num_nodes,**ikw)
    ind_mp = torch.empty(num_nodes,**ikw)
    for i in range(world_size):
        iid = all_ids[i]
        if i != rank:
            ex_shared_ids = torch.ones(num_nodes,dtype= torch.bool)
            ex_shared_ids[ids] = False
            ex_shared_ids = ex_shared_ids[iid]
            ex_id = iid[ex_shared_ids.nonzero().reshape(-1)]
            part_mp[ex_id] = i
            ind_mp[ex_id] = torch.arange(all_ids[i].size(0),**ikw)[ex_shared_ids]
        else:
            part_mp[iid] = i
            ind_mp[iid] = torch.arange(all_ids[i].size(0),**ikw)
    return DistIndex(ind_mp,part_mp).dist,shared_id_index
def build_drop_index(ids,e_src,e_dst,num_nodes,local_mask):
    ctx = _get_default_dist_context()
    ikw = dict(dtype=torch.long, device=torch.device('cpu'))
    not_drop = torch.zeros(num_nodes,dtype=torch.bool,device=ikw['device'])
    not_drop[ids] = True
    dist.all_reduce(not_drop, op=dist.ReduceOp.MAX,group=ctx.gloo_group)
    drop_eids = (~not_drop).nonzero().reshape(-1)
    drop_e_src = e_src[~not_drop]
    drop_e_dst = e_dst[~not_drop]
    return drop_eids[local_mask[drop_e_src]]
    #return drop_eids[local_mask[drop_e_src]|local_mask[drop_e_dst]]
    #random_part = torch.randint_like(drop_eids,low = 0,high = ctx.memory_group_size)
    #dist.broadcast(random_part,src=0,group=ctx.memory_gloo_group)
    #return drop_eids[random_part == ctx.memory_group_rank]

def build_index(ids):
    ctx = _get_default_dist_context()
    rank = ctx.memory_group_rank
    world_size = ctx.memory_group_size
    ikw = dict(dtype=torch.long, device=torch.device('cpu'))
    num_nodes = torch.zeros(1, **ikw)
    num_nodes[0] = ids.max().item()+1
    dist.all_reduce(num_nodes, op=dist.ReduceOp.MAX,group=ctx.gloo_group)
    all_ids: List[torch.Tensor] = [None] * world_size
    shared_id_index:  List[torch.Tensor] = [None] * world_size
    dist.all_gather_object(all_ids,ids,group=ctx.memory_gloo_group)
    part_mp = torch.empty(num_nodes,**ikw)
    ind_mp = torch.empty(num_nodes,**ikw)
    is_drop = torch.ones(num_nodes,dtype=torch.bool,device=ikw['device'])
    is_drop[ids] = False
    print('num nodes is {}'.format(num_nodes))
    for i in range(world_size):
        iid = all_ids[i].reshape(-1)
        part_mp[iid] = i
        ind_mp[iid] = torch.arange(all_ids[i].shape[0],**ikw)
        is_drop[iid] = False
        if i != rank:
            ex_shared_ids = is_drop[iid]
            ex_id = iid[ex_shared_ids.nonzero().reshape(-1)]
            part_mp[ex_id] = i
            ind_mp[ex_id] = torch.arange(all_ids[i].size(0),**ikw)[ex_shared_ids]
        else:
            part_mp[iid] = i
            ind_mp[iid] = torch.arange(all_ids[i].size(0),**ikw)

    return DistIndex(ind_mp,part_mp).dist

from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
def load_from_shared_node_partition(data,node_i,shared_node,sample_add_rev = True,edge_i = None, reid = None,device = torch.device('cuda'),feature_device = torch.device('cuda'),cached_full_feature=False):
    ctx = _get_default_dist_context()
    src, dst, ts, ext_roll = load_edge_tensors(data)
    print('tot edge {} same edge {}\n'.format(
        src.shape[0], (src == dst).sum().item()
    ))
    train_mask = (ext_roll == 0)
    val_mask = (ext_roll == 1)
    test_mask = (ext_roll == 2)
    num_node = max(src.max().item(),dst.max().item())+1 
    dim_feats = [0, 0, 0, 0, 0, 0, 0, 0]
    if ctx.local_rank == 0:
        _node_feats, _edge_feats = load_feat(data, num_node, src.shape[0], 172, 172)
        if _node_feats is not None:
            dim_feats[0] = _node_feats.shape[0]
            dim_feats[1] = _node_feats.shape[1]
            dim_feats[2] = _node_feats.dtype
            dim_feats[6] = 1

            node_feats = create_shared_mem_array('node_feats', _node_feats.shape, dtype=_node_feats.dtype)
            print('memory used is {} {} {}\n'.format(_node_feats.shape,_node_feats.dtype,_node_feats.numel()*_node_feats.element_size()/1024**3))
            print('dist rank is {} after node feats defination:'.format(torch.distributed.get_rank()))
            node_feats.copy_(_node_feats)
            del _node_feats
        else:
            node_feats = None
        if _edge_feats is not None:
            dim_feats[3] = _edge_feats.shape[0]
            dim_feats[4] = _edge_feats.shape[1]
            dim_feats[5] = _edge_feats.dtype
            dim_feats[7] = 1
            edge_feats = create_shared_mem_array('edge_feats', _edge_feats.shape, dtype=_edge_feats.dtype)
            edge_feats.copy_(_edge_feats)
            del _edge_feats
        else: 
            edge_feats = None
    torch.distributed.barrier()
    torch.distributed.broadcast_object_list(dim_feats, src=0)
    print('dist rank is {} after node feats defination:'.format(torch.distributed.get_rank()))
    print()
    if ctx.local_rank > 0:
        node_feats = None
        edge_feats = None
        if dim_feats[6] == 1:
            node_feats = get_shared_mem_array('node_feats', (dim_feats[0], dim_feats[1]), dtype=dim_feats[2])
        if dim_feats[7] == 1:
            edge_feats = get_shared_mem_array('edge_feats', (dim_feats[3], dim_feats[4]), dtype=dim_feats[5])
    if reid is not None:
        src = src[reid]
        dst = dst[reid]
        ts = ts[reid]
        train_mask = train_mask[reid]
        val_mask = val_mask[reid]
        test_mask = test_mask[reid]
        edge_feats = None if edge_feats is None else edge_feats[reid]
    if sample_add_rev:
        sampler_full_graph = {'edge_index': torch.cat((torch.cat((src.reshape(-1,1),dst.reshape(-1,1)),dim=1).reshape(1,-1),
                                                   torch.cat((dst.reshape(-1,1),src.reshape(-1,1)),dim=1).reshape(1,-1)),dim=0),'ts':ts.reshape(-1,1).repeat(1,2).reshape(-1),
                                                   'train_mask':train_mask,'val_mask':val_mask,'test_mask':test_mask,'eids':torch.arange(src.shape[0]).reshape(-1,1).repeat(1,2).reshape(-1),
                                                   }
    else:
        sampler_full_graph = {'edge_index':torch.stack((src,dst)),'ts':ts,
                                'train_mask':train_mask,'val_mask':val_mask,'test_mask':test_mask,'eids':torch.arange(src.shape[0])}
    
    if node_i is None:
        node_i = torch.arange(num_node)
        shared_node = torch.tensor([],dtype=torch.int)
        edge_i = torch.arange(src.shape[0])
    local_node_mask = torch.zeros(num_node,dtype=torch.bool)
    local_node_mask[node_i] = True
    is_shared_node = torch.zeros(num_node,dtype=torch.bool)
    is_shared_node[shared_node] = True
    local_node_mask[shared_node] = True
    node_i = local_node_mask.nonzero().reshape(-1)
    #unique_node_mask = (local_node_mask & (~is_shared_node))
    #unique_local_edge = (local_node_mask[src] & local_node_mask[dst] &(unique_node_mask[src] | unique_node_mask[dst]))
    #local_shared_edge = (is_shared_node[src] & is_shared_node[dst]).nonzero().reshape(-1)
    #random_part = torch.randint_like(local_shared_edge,low = 0,high = ctx.memory_group_size)
    #dist.broadcast(random_part,src=0,group=ctx.memory_gloo_group)
    #unique_local_edge[local_shared_edge[random_part ==  ctx.memory_group_rank]] = True
    eids,_ = edge_i.sort()#unique_local_edge.nonzero().reshape(-1)
    if cached_full_feature :
        whole_node_feature = node_feats
        whole_edge_feature = edge_feats
    else:
        whole_node_feature = None
        whole_edge_feature = None
    drop_local_eids = build_drop_index(eids,src,dst,src.shape[0],local_node_mask)
    """
    """
    #eids=torch.cat(eids)
    print('local node num {} ,local edge num {}\n'.format(node_i.shape[0],eids.shape[0]))
    edge_index = torch.stack((src[eids],dst[eids]))
    ts = ts[eids]
    full_train_mask = train_mask
    train_mask = train_mask[eids]
    test_mask = test_mask
    val_mask = val_mask

    if node_feats is not None:
        node_feats = node_feats[node_i]
    if edge_feats is not None:
        edge_feats = torch.cat((edge_feats[eids],edge_feats[drop_local_eids]),dim = 0)
    #记录其他shared node在其他分区的索引
    to_shared_node = torch.zeros(num_node,dtype=torch.int32)
    to_shared_node[node_i] = torch.arange(node_i.shape[0],dtype=torch.int32)
    to_shared_node = to_shared_node[shared_node]

    all_node_dist_mapper,shared_nids_list = build_shared_index(node_i, shared_node,to_shared_node)
    all_node_dist_mapper[shared_node] = DistIndex(all_node_dist_mapper[shared_node]).set_shared()

    all_edge_dist_mapper = build_index(torch.cat((eids,drop_local_eids)))#build_index(eids)#torch.cat((eids,drop_local_eids)))
#    print('{} {}'.format(DistIndex(all_edge_dist_mapper).loc[DistIndex(all_edge_dist_mapper).part == ctx.memory_group_rank].max(),edge_feat.shape))
    dist_data_loader = DistributedGraphStore(node_i, eids,edge_index,ts,shared_node,nids_mapper = all_node_dist_mapper,\
                                             eids_mapper = all_edge_dist_mapper,shared_nids_list = shared_nids_list,nfeat = node_feats,efeat = edge_feats,device=device,feature_device=feature_device,use_pin=True,whole_node_feature=whole_node_feature,whole_edge_feature=whole_edge_feature)
    #dist_data_loader = DistributedGraphStore(node_i, eids,edge_index,ts,shared_node,nids_mapper = all_node_dist_mapper,\
    #                                     eids_mapper = all_edge_dist_mapper,shared_nids_list = shared_nids_list,nfeat = node_feat,efeat = edge_feat,device=device,feature_device=feature_device,use_pin=True)
    historical_cache_index = CachePushRoute(node_i.shape[0],torch.stack((src[drop_local_eids],dst[drop_local_eids])),all_node_dist_mapper)
    print('init data loader')
    return dist_data_loader,sampler_full_graph,train_mask,val_mask,test_mask,full_train_mask,historical_cache_index


def get_eval_batch(eid,nid_mapper,eid_mapper,batch_size):
    ctx = _get_default_dist_context()
    belong_batch = torch.arange(eid.shape[0],device=torch.device('cuda'),dtype=torch.long)
    belong_batch = belong_batch // batch_size
    max_batch = belong_batch[-1]
    print(max_batch,eid.shape[0])
    local_edge = (DistIndex(eid_mapper).part == ctx.memory_group_rank)
    global_edge = torch.zeros(dist.get_world_size(),max_batch+1)
    for i in range(dist.get_world_size()):
        mask = (DistIndex(eid_mapper).part == i)
        eid_for_test = mask[eid.to('cpu')].to('cuda')
        local_belong_batch = belong_batch[eid_for_test]
        pos = torch.arange(eid_for_test.sum().item(),device=torch.device('cuda'),dtype=torch.long)
        posl,_ = torch_scatter.scatter_min(pos,local_belong_batch,dim_size=max_batch+1)
        posr,_ = torch_scatter.scatter_max(pos,local_belong_batch,dim_size=max_batch+1)
        global_edge[i] = posr-posl+1

    print('average: {}\n'.format((global_edge.max(dim = 0)[0]/global_edge.min(dim=0)[0]).float().mean()))
    print('max average: {}\n'.format((global_edge.max(dim = 0)[0]).float().mean()))
    print('min average: {}\n'.format((global_edge.min(dim = 0)[0]).float().mean()))
    eid_for_test = local_edge[eid.to('cpu')].to('cuda')
    local_belong_batch = belong_batch[eid_for_test]
    pos = torch.arange(eid_for_test.sum().item(),device=torch.device('cuda'),dtype=torch.long)
    #posl = torch.ones(max_batch,device=torch.device('cuda'),dtype=torch.long)*eid_for_test.sum().item()
    posl,_ = torch_scatter.scatter_min(pos,local_belong_batch)
    #posr = torch.zeros(max_batch,device=torch.device('cuda'),dtype=torch.long) 
    posr,_ = torch_scatter.scatter_max(pos,local_belong_batch)
    return posl,posr,eid_for_test
    


def _load_partition_ids(path, *, optional=False):
    if not os.path.exists(path):
        if optional:
            return torch.empty(0, dtype=torch.long)
        raise FileNotFoundError(
            'partition ID file does not exist: {}'.format(path)
        )
    values = []
    with open(path, 'r') as source:
        for line_number, line in enumerate(source, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                values.append(int(value))
            except ValueError as error:
                raise ValueError(
                    'invalid ID at {}:{}'.format(path, line_number)
                ) from error
    return torch.tensor(values, dtype=torch.long)


def load_from_speed(data,seed,top,sampler_graph_add_rev,device=torch.device('cuda'),feature_device=torch.device('cuda'),partition_path='ours',):
    ctx = _get_default_dist_context()
    if ctx.memory_group_size == 1:
        return load_from_shared_node_partition(data,None,None,sample_add_rev=sampler_graph_add_rev,device=device,feature_device=feature_device)
    else:
        node_path = '{}/{}/{}/{}_{}parts_top{}/output{}.txt'.format(partition_path,data,seed,data,ctx.memory_group_size,top,ctx.memory_group_rank)
        shared_path = '{}/{}/{}/{}_{}parts_top{}/outputshared.txt'.format(partition_path,data,seed,data,ctx.memory_group_size,top)
        edge_path = '{}/{}/{}/{}_{}parts_top{}/edge_output{}.txt'.format(partition_path,data,seed,data,ctx.memory_group_size,top,ctx.memory_group_rank)
        node_i = _load_partition_ids(node_path)
        shared_node = _load_partition_ids(shared_path, optional=True)
        edge_i = _load_partition_ids(edge_path)
      
        return load_from_shared_node_partition(data,node_i,shared_node,sample_add_rev=sampler_graph_add_rev,edge_i=edge_i,reid=None,device=device,feature_device=feature_device)



