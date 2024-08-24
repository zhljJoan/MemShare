from typing import List, Optional
import starrygl
from starrygl import distributed
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor
from starrygl.sample.graph_core.utils import _get_pin, build_mapper
import os.path as osp
import torch
import torch.distributed as dist
from torch_geometric.data import Data

from starrygl.utils.uvm import *
class DistributedGraphStore:
    
    def __init__(self, ids, eids, edge_index, ts,
                shared_nids = None, nids_mapper=None, eids_mapper = None,shared_nids_list = None, 
                nfeat = None, efeat = None, pre_nfeat = None, pre_efeat = None,
                is_cached_1_hop = False, is_cache_index_dynmaic = False, device = torch.device('cuda'),feature_device = torch.device('cuda'),
                use_pin = False,whole_node_feature = None, whole_edge_feature = None):
        ctx = distributed.context._get_default_dist_context()
        print('init {}'.format(ctx.local_rank))
        self.device = device if device.type == 'cpu' else torch.device('cuda:{}'.format(ctx.local_rank))
        self.feature_device = feature_device if feature_device.type == 'cpu' else torch.device('cuda:{}'.format(ctx.local_rank))
        print(self.feature_device)
        self.ids = ids
        self.nids_mapper = nids_mapper
        self.eids = eids
        self.ts = ts
        self.eids_mapper = eids_mapper
        self.edge_index = edge_index
        self.shared_nids = shared_nids
        self.shared_nids_list = shared_nids_list
        self.num_nodes = self.nids_mapper.shape[0]
        print(self.eids_mapper)
        self.use_pin = use_pin
        self.num_edges = self.eids_mapper.shape[0]
        self._nfeat_pins = {}
        self._efeat_pins = {}
        if torch.cuda.is_available():
            print("Total GPU memory: ", torch.cuda.get_device_properties(0).total_memory/1024**3)
            print("Current GPU memory allocated: ", torch.cuda.memory_allocated(0)/1024**3)
            print("Current GPU memory reserved: ", torch.cuda.memory_reserved(0)/1024**3)
            print("Max GPU memory allocated during this session: ", torch.cuda.max_memory_allocated(0))
            print("Max GPU memory reserved during this session: ", torch.cuda.max_memory_reserved(0))
        else:
            print("CUDA is not available.")
    
        print(feature_device,ctx.local_rank)
        """
        if(boundery_probability > 0 and boundery_edges is not None):
            add_node_dim = min(int(boundery_edges.shape[0] * boundery_probability)*2,boundery_nodes)
            nfeat0 = torch.zeros(self.ids.shape[0]+add_node_dim,nfeat.shape[1])
            nfeat0[:self.ids.shape[0]] = nfeat
            nfeat= nfeat0
        """
        self.nfeat = DistributedTensor(nfeat.to(self.feature_device)) if nfeat is not None else None
        if torch.cuda.is_available():
            print("Total GPU memory: ", torch.cuda.get_device_properties(0).total_memory/1024**3)
            print("Current GPU memory allocated: ", torch.cuda.memory_allocated(0)/1024**3)
            print("Current GPU memory reserved: ", torch.cuda.memory_reserved(0)/1024**3)
            print("Max GPU memory allocated during this session: ", torch.cuda.max_memory_allocated(0))
            print("Max GPU memory reserved during this session: ", torch.cuda.max_memory_reserved(0))
        else:
            print("CUDA is not available.")
        """
        if(boundery_probability > 0 and boundery_edges is not None):
            add_edge_dim = int(boundery_edges.shape[0] * boundery_probability)
            efeat0 = torch.zeros(self.eids.shape[0]+add_edge_dim,efeat.shape[1])
            efeat0[efeat.shape[0]] = efeat
            efeat= efeat0
        """
        self.efeat = DistributedTensor(efeat.to(self.feature_device)) if efeat is not None else None
        print(efeat.shape)
        if torch.cuda.is_available():
            print("Total GPU memory: ", torch.cuda.get_device_properties(0).total_memory/1024**3)
            print("Current GPU memory allocated: ", torch.cuda.memory_allocated(0)/1024**3)
            print("Current GPU memory reserved: ", torch.cuda.memory_reserved(0)/1024**3)
            print("Max GPU memory allocated during this session: ", torch.cuda.max_memory_allocated(0))
            print("Max GPU memory reserved during this session: ", torch.cuda.max_memory_reserved(0))
        else:
            print("CUDA is not available.")
        self.whole_node_feature = whole_node_feature
        self.whole_edge_feature = whole_edge_feature
        """
        self.is_cached_1_hop = is_cached_1_hop
        self.is_cache_index_dynmaic = is_cache_index_dynmaic
        self.use_pin = use_pin
        self.boundery_edges = boundery_edges
        self.boundery_nodes = boundery_nodes
        self.boundery_nodes_feature = boundery_nodes_feature
        self.boundery_edges_feature = boundery_edges_feature 
        self.drop_nid = None
        self.drop_eid = None
        """
    """
    def fetch_boundery_edge(self,probability):
        add_edge = torch.randint(0,self.boundery_edges.shape[0],int(probability*self.boundery_edges)).unique()
        add_edge_id = self.eids_mapper[self.boundery_edges[2,add_edge]]
        add_edge_src = self.boundery_edges[0,add_edge]
        add_edge_dst = self.boundery_edges[1,add_edge]
        add_node_id = self.boundery_nodes[torch.cat((add_edge_src,add_edge_dst))]
        add_edge_mask = DistIndex(self.eids_mapper[add_edge_id]) != dist.get_rank()
        add_node_mask = DistIndex(self.nids_mapper[add_node_id]) != dist.get_rank()
        num_add_edges = add_edge_mask.sum()
        num_add_nodes = add_node_mask.sum()
        self.efeat.accessor.data[self.eids.shape[0]:self.eids.shape[0]+num_add_edges] = self.boundery_edges_feature[add_edge_mask.nonzero().reshape(-1)].continues().to(self.feature_device)
        self.nfeat.accessor.data[self.ids.shape[0]:self.ids.shape[0]+num_add_nodes] = self.boundery_node_feature[add_node_id[add_node_mask]].continues().to(self.feature_device)
        self.nids_mapper[add_node_id[add_node_mask]] = torch.arange(self.ids.shape[0],self.ids.shape[0]+num_add_nodes)
        self.eids_mapper[add_edge_id[add_edge_mask]] =  torch.arange(self.eids.shape[0],self.eids.shape[0]+num_add_edges)
    """

    def get_dist_index(self,ind,mapper):
        return mapper[ind.to(mapper.device)]

    def get_dist_nfeat(self,idx = None, 
                       send_ptr: Optional[List[int]] = None,
                       recv_ptr: Optional[List[int]] = None,
                       recv_ind: Optional[List[torch.Tensor]] = None,is_async=False,
                       group = None,out_device =None):
        if self.nfeat is None:
            return None
        else:
            if self.nfeat.device.type == 'cpu' and self.use_pin:
                 pin_func = self._get_nfeat_pin
                 return self.nfeat.all_to_all_get(idx,send_ptr,recv_ptr,recv_ind,is_async,group,pin_func,out_device)
            else:
                return self.nfeat.all_to_all_get(idx,send_ptr,recv_ptr,recv_ind,is_async,group)

    def get_local_nfeat(self,idx,out_device = torch.device('cuda')):
        if self.nfeat is None:
            return None
        else:
            idx = DistIndex(idx).loc.to(self.nfeat.device)
            rows = idx.shape[0]
            if self.nfeat.device.type == 'cpu' and self.use_pin : 
                nfeat_pin = self._get_nfeat_pin(0,rows)
                torch.index_select(self.nfeat.accessor.data,0,idx,out = nfeat_pin)
                return nfeat_pin.to(out_device,non_blocking = True)
            else:
                return self.nfeat.accessor.data[idx].to(out_device)
            
    def get_local_nfeat_from_whole(self,idx,out_device = torch.device('cuda')):
            rows = idx.shape[0]
            if self.nfeat.device.type == 'cpu' and self.use_pin : 
                nfeat_pin = self._get_nfeat_pin(0,rows)
                torch.index_select(self.whole_node_feature,0,idx,out = nfeat_pin)
                return nfeat_pin.to(out_device,non_blocking = True)
            else:
                return self.whole_node_feature.to('cpu')
    
    def get_dist_efeat(self,idx = None,
                        send_ptr: Optional[List[int]] = None,
                       recv_ptr: Optional[List[int]] = None,
                       recv_ind: Optional[List[torch.Tensor]] = None,is_async=False,is_sorted=False,
                       group = None,out_device = None):
        if self.efeat is None:
            return None
        elif is_async == False:
            idx,pos = idx.sort()
            xpos = torch.empty_like(pos)
            xpos[pos] = torch.arange(pos.shape[0],device = pos.device,dtype=pos.dtype)
            if self.efeat.device.type == 'cpu' and self.use_pin:
                pin_func = self._get_efeat_pin
                return self.efeat.all_to_all_get(idx,send_ptr,recv_ptr,recv_ind,is_async,group,pin_func,out_device)[xpos]
            else:
                return self.efeat.all_to_all_get(idx,send_ptr,recv_ptr,recv_ind,is_async,group)[xpos]
        else:
            if self.efeat.device.type == 'cpu' and self.use_pin:
                pin_func = self._get_efeat_pin
                return self.efeat.all_to_all_get(idx,send_ptr,recv_ptr,recv_ind,is_async,group,pin_func,out_device)
            else:
                return self.efeat.all_to_all_get(idx,send_ptr,recv_ptr,recv_ind,is_async,group)
        
    def get_local_efeat(self,idx,out_device = torch.device('cuda')):
        if self.efeat is None:
            return None
        else:
            idx = DistIndex(idx).loc.to(self.efeat.device)
            rows = idx.shape[0]
            if self.efeat.device.type == 'cpu' and self.use_pin : 
                efeat_pin = self._get_efeat_pin(0,rows)
                torch.index_select(self.efeat.accessor.data,0,idx,out=efeat_pin)
                return efeat_pin.to(out_device,non_blocking = True)
            else:
                return self.efeat.accessor.data[idx.to(self.efeat.device)].to(out_device)
    
    def get_local_efeat_from_whole(self,idx,out_device = torch.device('cuda')):
        rows = idx.shape[0]
        if self.efeat.device.type == 'cpu' and self.use_pin : 
            efeat_pin = self._get_efeat_pin(0,rows)
            torch.index_select(self.whole_edge_feature,0,idx,out=efeat_pin)
            return efeat_pin.to(out_device,non_blocking = True)
        else:
            return self.whole_edge_feature[idx.to('cpu')].to(out_device)
        
    def get_nfeat(self,idx_dict,use_dist = True,group = None,is_async = False,out_device = torch.device('cuda')):
        if use_dist:
            return self.get_dist_nfeat(**idx_dict,group=group,is_async=is_async,out_device=out_device)
        else:
            return self.get_local_nfeat(idx_dict['idx'],out_device=out_device)
        
    def get_efeat(self,idx_dict,use_dist = True,group = None,is_async = False, out_device = torch.device('cuda')):
        if use_dist:
            return self.get_dist_nfeat(**idx_dict,group=group,is_async=is_async,out_device=out_device)
        else:
            return self.get_local_nfeat(idx_dict['idx'],out_device=out_device)
        
    def _get_efeat_pin(self, layer: int, rows: int) -> torch.Tensor:
        return _get_pin(self._efeat_pins, layer, rows, self.efeat.shape[1:])

    def _get_nfeat_pin(self, layer: int, rows: int) -> torch.Tensor:
        return _get_pin(self._nfeat_pins, layer, rows, self.nfeat.shape[1:])

    '''
    def __init__(self, pdata, device = torch.device('cuda'),
                 uvm_node = False, 
                 uvm_edge = False,
                 use_cache = False,
                 feature_device = torch.device('cuda'),
                 full_sample_graph = True):
        self.device = device
        self.ids = pdata.ids.to(device)
        self.eids = pdata.eids.to(device)
        self.edge_index = pdata.edge_index.to(device)
        current_device = torch.cuda.current_device()
        # 打印设备名称
        print('Device:', current_device)

    # 打印当前设备的显存使用情况（单位：字节）
        print('init Memory Usage:')
        print('init Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
        print('init Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB')
        print('ids mem is {} eids mem is {} edge_inde mem is {}\n'.format(self.ids.numel()*self.ids.element_size()/1024/1024/1024,
                                                     self.eids.numel()*self.eids.element_size()/1024/1024/1024,
                                                     self.edge_index.numel()*self.edge_index.element_size()/1024/1024/1024,))
        if hasattr(pdata,'edge_ts'):
            self.edge_ts = pdata.edge_ts.to(device)#.to(torch.float)
        else:
            self.edge_ts = None
        self.sample_graph = pdata.sample_graph
        self.nids_mapper = build_mapper(nids=pdata.ids.to(device)).dist.to('cpu')
        self.eids_mapper = build_mapper(nids=pdata.eids.to(device)).dist.to('cpu')
        if full_sample_graph is False:
            local_sample_mask = ((DistIndex(self.nids_mapper[self.sample_graph['edge_index'][0,:]]).part == dist.get_rank()) | (DistIndex(self.nids_mapper[self.sample_graph['edge_index'][1,:]]).part == dist.get_rank()))
            self.sample_graph['edge_index'] = self.sample_graph['edge_index'][:,local_sample_mask]
            for k in self.sample_graph:
                if isinstance(self.sample_graph[k],torch.Tensor) and k != 'edge_index':
                    self.sample_graph[k] = self.sample_graph[k][local_sample_mask]
        self.sample_graph['dist_eid'] = self.eids_mapper[pdata.sample_graph['eids']]
        for k in self.sample_graph:
            if isinstance(self.sample_graph[k],torch.Tensor):
                self.sample_graph[k] = self.sample_graph[k]#.to(device)
                print('{} memory is {}'.format(k,self.sample_graph[k].numel()*self.sample_graph[k].element_size()/1024/1024/1024))
        torch.cuda.empty_cache()
        # 打印设备名称
        print('mapper Memory Usage:')
        print('mapper Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
        print('mapper Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB')
        self.num_nodes = self.nids_mapper.data.shape[0]
        self.num_edges = self.eids_mapper.data.shape[0]
        world_size = dist.get_world_size()
        self.uvm_node = uvm_node
        self.uvm_edge = uvm_edge
        self.feature_device = feature_device if feature_device.type=='cpu' else torch.cuda.current_device()
        if hasattr(pdata,'x') and pdata.x is not None:
            ctx = DistributedContext.get_default_context()
            pdata.x = pdata.x.to(torch.float)
            if uvm_node == False :
                x = pdata.x.to(feature_device)
            else:
                if feature_device.type == 'cuda':
                    x = uvm_empty(*pdata.x.size(),
                                    dtype=pdata.x.dtype,
                                    device=ctx.device)
                    uvm_share(x,device = ctx.device)
                    uvm_advise(x,cudaMemoryAdvise.cudaMemAdviseSetAccessedBy)
                    uvm_prefetch(x)
            if world_size > 1:
                self.x = DistributedTensor(pdata.x.to(feature_device).to(torch.float))
            else:
                self.x = x
        else:
            self.x = None
        print('x Memory Usage:')
        print('x Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
        print('x Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB') 
        if hasattr(pdata,'edge_attr') and pdata.edge_attr is not None:
            ctx = DistributedContext.get_default_context()
            pdata.edge_attr = pdata.edge_attr.to(torch.float16)
            if uvm_edge == False :
                edge_attr = pdata.edge_attr.to(feature_device)
            else:
                if feature_device.type == 'cuda':
                    edge_attr = uvm_empty(*pdata.edge_attr.size(),
                                    dtype=pdata.edge_attr.dtype,
                                    device=ctx.device)
                    edge_attr = uvm_share(edge_attr,device = torch.device('cpu'))
                    edge_attr.copy_(pdata.edge_attr)

                    edge_attr = uvm_share(edge_attr,device = ctx.device)
                    uvm_advise(edge_attr,cudaMemoryAdvise.cudaMemAdviseSetAccessedBy)
                    uvm_prefetch(edge_attr)
                    
            if world_size > 1:
                self.edge_attr = DistributedTensor(edge_attr)
            else:
                self.edge_attr = edge_attr
        else:
            self.edge_attr = None
#        print('edge mem is {} node mem is {}'.format(self.edge_attr.accessor.data.numel()*self.edge_attr.accessor.data.element_size()/1024/1024/1024,
#                                                     self.x.accessor.data.numel()*self.x.accessor.data.element_size()/1024/1024/1024))
        print('edge Memory Usage:')
        print('edge Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
        print('edge Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB')
        if use_cache is True:
            self.cache_node_index,self.cache_node_attr = self._build_cache_node_feature()
            self.cache_node_mapper = torch.zeros_like(self.nids_mapper,device=device)
            self.cache_node_mapper[self.cache_node_index] = torch.arange(self.cache_node_index.shape[0],device=device)
            print('cache node Memory Usage:')
            print('cache node Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
            print('cache node Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB')
        #    print('cache node cache mem is {}'.format(self.cache_node_attr.numel()*self.cache_node_attr.element_size()/1024/1024/1024))
            if self.edge_attr is not None:
                self.cache_edge_index,self.cache_edge_attr = self._build_cache_edge_feature()
                self.cache_edge_mapper = torch.zeros_like(self.eids_mapper,device=device)
                self.cache_edge_mapper[self.cache_edge_index] = torch.arange(self.cache_edge_index.shape[0],device=device)
            else:
                self.cache_edge_attr = None
        print('all Memory Usage:')
        print('all Allocated:', round(torch.cuda.memory_allocated(current_device)/1024**3,1), 'GB')
        print('all Cached:   ', round(torch.cuda.memory_reserved(current_device)/1024**3,1), 'GB') 
    '''
    
    
    #def _get_node_attr(self,ids,asyncOp = False):
    #    '''
    #    Retrieves node attributes for the specified node IDs.
#
    #    Args:
    #        ids: Node IDs for which to retrieve attributes.
#
    #        asyncOp: If True, performs asynchronous operation for distributed data.
#
    #    '''
    #    if self.x is None:
    #        return None
    #    elif dist.get_world_size() == 1:
    #        return self.x[ids]
    #    else:
    #        if self.x.rrefs is None or asyncOp is False:
    #            ids = self.x.all_to_all_ind2ptr(ids)
    #            return self.x.all_to_all_get(**ids)
    #        return self.x.index_select(ids)
    #def _get_edge_attr(self,ids,asyncOp = False):
    #    '''
    #    Retrieves edge attributes for the specified edge IDs.
#
    #    Args:
    #        ids: Edge IDs for which to retrieve attributes.
#
    #        asyncOp: If True, performs asynchronous operation for distributed data.
#
    #    '''
    #    if self.edge_attr is None:
    #        return None
    #    elif dist.get_world_size() == 1:
    #        return self.edge_attr[ids]
    #    else:
    #        if self.edge_attr.rrefs is None or asyncOp is False:
    #            ids = self.edge_attr.all_to_all_ind2ptr(ids)
    #            return self.edge_attr.all_to_all_get(**ids)
    #        return self.edge_attr.index_select(ids)
    #
    #def _get_dist_index(self,ind,mapper):
    #    '''
    #    Retrieves the distributed index for the specified local index using the provided mapper.
#
    #    Args:
    #        ind: Local index for which to retrieve the distributed index.
#
    #        mapper: Mapper providing the distributed index.
#
    #    '''
#
#
    #    return mapper[ind.to(mapper.device)]
#
    #def _build_cache_node_feature(self):
    #    src = self.edge_index[0,:]
    #    dist_src = DistIndex(self.nids_mapper[src.to('cpu')].to('cuda')).dist
    #    dist_src,ind = dist_src.sort()
    #    dst = DistributedTensor(self.edge_index[1,:])
    #    send_ptr = dst.all_to_all_ind2ptr(dist_src,ind)
    #    dst_neighbor = dst.all_to_all_send(**send_ptr)
    #    nids = torch.cat((self.ids,dst_neighbor))
    #    nids = nids.unique()
    #    dist_index = DistIndex(self.nids_mapper[nids.to('cpu')].to('cuda').sort()[0])
    #    local_mask = dist_index.part == dist.get_rank()
    #    cache_index = nids[~local_mask]
    #    print(cache_index.shape)
    #    if self.x is not None:
    #        attr = self._get_node_attr(dist_index.dist[~local_mask])
    #        return cache_index,attr
    #    return cache_index,None
    #
    #def _get_cache_node_attr(self,ids,mask=None):
    #    if self.cache_node_attr is None:
    #        return None
    #    dist_id = DistIndex(self._get_dist_index(ids,self.nids_mapper))
    #    if mask is None:
    #        mask = dist_id.part == dist.get_rank()
    #    return torch.cat((self.x[dist_id.loc[mask]],self.cache_node_attr[self.cache_node_mapper[ids[~mask]].to(self.cache_edge_attr.device)]),dim = 0)
    #
    #def _get_cache_edge_attr(self,ids,mask=None):
    #    if self.cache_edge_attr is None:
    #        return None
    #    dist_id = DistIndex(self._get_dist_index(ids,self.eids_mapper))
    #    if mask is None:
    #        mask = dist_id.part == dist.get_rank()
#
    #    tensor = torch.cat((self.edge_attr[dist_id.loc[mask]],self.cache_edge_attr[self.cache_edge_mapper[ids[~mask]].to(self.cache_edge_attr.device)]),dim = 0)
    #    return tensor
    #def _build_cache_edge_feature(self):
    #    src = self.edge_index[1,:]
    #    dist_src = DistIndex(self.nids_mapper[src.to('cpu')].to('cuda')).dist
    #    dist_src,ind = dist_src.sort()
    #    eid = DistributedTensor(self.eids)
    #    send_ptr = eid.all_to_all_ind2ptr(dist_src,ind)
    #    eid_neighbor = eid.all_to_all_send(**send_ptr)
    #    eids = torch.cat((self.eids,eid_neighbor))
    #    eids = eids.unique()
    #    dist_index = DistIndex(self.eids_mapper[eids.to('cpu')].to('cuda').sort()[0]) 
    #    local_mask = dist_index.part == dist.get_rank()
    #    cache_index = eids[~local_mask]
    #    print('the cache shape is {} the all shape is {}\n'.format(cache_index.shape,self.eids.shape))
    #    attr = self._get_edge_attr(dist_index.dist[~local_mask])
    #    return cache_index,attr
    

class DataSet:
    '''

    Args:
        nodes: Tensor representing nodes. If not None, it is moved to the specified device.

        edges: Tensor representing edges. If not None, it is moved to the specified device.

        labels: Optional parameter for labels.

        ts: Tensor representing timestamps. If not None, it is moved to the specified device.

        device: Device to which tensors are moved (default is 'cuda').

    '''
    def __init__(self,nodes = None,
                 edges = None,
                 labels = None, 
                 ts = None, 
                 device = torch.device('cuda'),**kwargs):
        if nodes is not None:
            self.nodes = nodes.to(device)
        if edges is not None:
            self.edges = edges.to(device)
        if ts is not None:
            self.ts = ts.to(device)
        if labels is not None:
            self.labels = labels
        
        for k, v in kwargs.items():
            assert (isinstance(v,torch.Tensor) or isinstance(v,DistributedTensor)) and v.shape[0]==self.len
            setattr(self, k, v.to(device))
    @property
    def len(self):
        return  self.nodes.shape[0] if hasattr(self,'nodes') else self.edges.shape[1]
     
    def _get_empty(self):
        '''
        Creates an empty dataset with the same device and data types as the current instance.

        '''
        nodes = torch.empty([],dtype = self.nodes.dtype,device= self.nodes.device)if hasattr(self,'nodes') else None
        edges = torch.empty([[],[]],dtype = self.edges.dtype,device= self.edge.device)if hasattr(self,'edges') else None
        d = DataSet(nodes,edges)
        for k,v in self.__dict__.items():
            if k == 'edges' or k=='nodes' or k == 'len':
                continue
            else:
                setattr(d,k,torch.empty([]))
        return d
    
    def __getitem__(self,indx):
        nodes = self.nodes[indx] if hasattr(self,'nodes') else None
        edges = self.edges[:,indx] if hasattr(self,'edges') else None
        d = DataSet(nodes,edges)
        for k,v in self.__dict__.items():
            if k == 'edges' or k=='nodes' or k == 'len':
                continue
            else:
                setattr(d,k,v[indx])
        return d
    @property
    def device(self):
        return self.edges.device if hasattr(self,'edges') else self.nodes.device

    #@staticmethod
    def get_next(self,indx):
        '''
        Retrieves the next dataset based on the provided index.

        Args:
            indx: Index specifying the dataset to retrieve.

        '''
        nodes = self.nodes[indx] if hasattr(self,'nodes') else None
        edges = self.edges[:,indx] if hasattr(self,'edges') else None
        d = DataSet(nodes,edges)
        for k,v in self.__dict__.items():
            if k == 'edges' or k=='nodes' or k == 'len':
                continue
            else:
                setattr(d,k,v[indx])
        return d

    #@staticmethod
    def shuffle(self):
        '''
        Shuffles the dataset and returns a new dataset with the same attributes.

        '''
        indx = torch.randperm(self.len)
        nodes = self.nodes[indx] if hasattr(self,'nodes') else None
        edges = self.edges[:,indx] if hasattr(self,'edges') else None
        d = DataSet(nodes,edges)
        for k,v in self.__dict__.items():
            if k == 'edges' or k=='nodes' or k == 'len':
                continue
            else:
                setattr(d,k,v[indx])
        return d
    @staticmethod
    def scatter_train_data(data,id_mapper):
        _,dst = data.edges
        dst_index,ind = id_mapper[dst.to('cpu')].to('cuda').sort()
        src = DistIndex(torch.arange(ind.shape[0],device = ind.device,dtype = torch.long),torch.full_like(ind,dist.get_rank()))
        train_dst_edges = torch.stack((src.dist,dst))
        train_dst_data = DataSet(edges = DistributedTensor(train_dst_edges.T))
        for k,v in data.__dict__.items():
            if k == 'edges' or k == 'nodes' or v == None:
                continue   

            train_dst_data.__setattr__(k,DistributedTensor(v))
        send_dict = train_dst_data.ts.all_to_all_ind2ptr(dst_index,send_index = ind)
        for k,v in train_dst_data.__dict__.items():
            train_dst_data.__setattr__(k,v.all_to_all_send(**send_dict)) 

        train_dst_data.edges = train_dst_data.edges.T
        return train_dst_data,send_dict
    
class TemporalGraphData(DistributedGraphStore):
    def __init__(self,pdata,device):
        super(DistributedGraphStore,self).__init__(pdata,device)
    def _set_temporal_batch_cache(self,size,pin_size):
        pass
    def _load_feature_to_cuda(self,ids):
        pass
    



class TemporalNeighborSampleGraph(DistributedGraphStore):
    '''

    Args:
        sample_graph: A dictionary containing graph structure information, including 'edge_index', 'ts' (edge timestamp), and 'eids' (edge identifiers).

        mode: Specifies the dataset mode ('train', 'val', 'test', or 'full').

        eids_mapper: Optional parameter for edge identifiers mapping.


    '''
    def __init__(self, sample_graph=None, mode='full', dist_eid_mapper = None,local_eids = None):
        if local_eids is not None:
            local_mask = torch.zeros(sample_graph['eids'].max().item() + 1,dtype=torch.bool)
            local_mask[local_eids] = True
            local_mask = local_mask[sample_graph['eids']]
            self.edge_index = sample_graph['edge_index'][:,local_mask].to('cpu')
            self.num_edges = self.edge_index.shape[1]
            self.edge_ts = sample_graph['ts'][local_mask].to('cpu')
            self.eid = sample_graph['eids'][local_mask]
        else:
            self.edge_index = sample_graph['edge_index'].to('cpu')
            self.num_edges = self.edge_index.shape[1]
            self.edge_ts = sample_graph['ts'].to('cpu')
            self.eid = sample_graph['eids']#torch.arange(self.num_edges,dtype = torch.long)
        self.dist_eid = dist_eid_mapper[self.eid]
        #sample_graph['eids']
        if mode == 'train':
            mask = sample_graph['train_mask'] 
        if mode == 'val':
            mask = sample_graph['val_mask']
        if mode == 'test':
            mask = sample_graph['test_mask']
        if mode != 'full':
            self.edge_index = self.edge_index[:, mask[self.eid]].to('cpu')
            self.edge_ts = self.edge_ts[mask[self.eid]].to('cpu')
            self.eid = self.eid[mask[self.eid]].to('cpu')  


        
    


