from starrygl.distributed.utils import DistIndex, DistributedTensor
from starrygl.sample.graph_core.utils import build_mapper
import os.path as osp
import torch
import torch.distributed as dist
from torch_geometric.data import Data
class DistributedGraphStore:
    def __init__(self, pdata, device = torch.device('cuda'), all_on_gpu = False):
        self.device = device
        self.ids = pdata.ids.to(device)
        self.edge_index = pdata.edge_index.to(device)
        if hasattr(pdata,'edge_ts'):
            self.edge_ts = pdata.edge_ts.to(device).to(torch.float)
        else:
            self.edge_ts = None
        self.sample_graph = pdata.sample_graph
        self.nids_mapper = build_mapper(nids=pdata.ids.to(device)).dist.to('cpu')
        self.eids_mapper = build_mapper(nids=pdata.eids.to(device)).dist.to('cpu')
        torch.cuda.empty_cache()
        if all_on_gpu:
            self.nids_mapper = self.nids_mapper.to(device)
            self.eids_mapper = self.eids_mapper.to(device)
        self.num_nodes = self.nids_mapper.data.shape[0]
        world_size = dist.get_world_size()
        if hasattr(pdata,'x') and pdata.x is not None:
            if world_size > 1:
                self.x = DistributedTensor(pdata.x.to(self.device).to(torch.float))
            else:
                self.x = pdata.x.to(device).to(torch.float)
        else:
            self.x = None
        if hasattr(pdata,'edge_attr') and pdata.edge_attr is not None:
            if world_size > 1:
                self.edge_attr = DistributedTensor(pdata.edge_attr.to('cuda').to(torch.float))
            else:
                self.edge_attr = pdata.edge_attr.to('cuda').to(torch.float)
        else:
            self.edge_attr = None

    def _get_node_attr(self,ids):
        if self.x is None:
            return None
        elif dist.get_world_size() == 1:
            return self.x[ids]
        else:
            return self.x.index_select(ids)
    def _get_edge_attr(self,ids,):
        if self.edge_attr is None:
            return None
        elif dist.get_world_size() == 1:
            return self.edge_attr[ids]
        else:
            return self.edge_attr.index_select(ids)


class DataSet:
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
        self.len = self.nodes.shape[0] if nodes is not None else self.edges.shape[1] 
        print(self.edges.shape,self.len)
        for k, v in kwargs.items():
            assert isinstance(v,torch.Tensor) and v.shape[0]==self.len
            setattr(self, k, v.to(device))
    def _get_empty(self):
        nodes = torch.empty([],dtype = self.nodes.dtype,device= self.nodes.device)if hasattr(self,'nodes') else None
        edges = torch.empty([[],[]],dtype = self.edges.dtype,device= self.edge.device)if hasattr(self,'edges') else None
        d = DataSet(nodes,edges)
        for k,v in self.__dict__.items():
            if k == 'edges' or k=='nodes' or k == 'len':
                continue
            else:
                setattr(d,k,torch.empty([]))
        return d

    #@staticmethod
    def get_next(self,indx):
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
    
class DistributedTemporalGraphData(DistributedGraphStore):
    def __init__(self,pdata,device):
        super(DistributedTemporalGraphData,self).__init__(pdata,device)
    def _set_temporal_batch_cache(self,size,pin_size):
        pass
    def _load_feature_to_cuda(self,ids):
        pass
    



class TemporalNeighborSampleGraph(DistributedGraphStore):
    def __init__(self, sample_graph=None, mode='full', eids_mapper=None):
        self.edge_index = sample_graph['edge_index']
        self.num_edges = self.edge_index.shape[1]
        if 'ts' in sample_graph:
            self.edge_ts = sample_graph['ts']
        else:
            self.edge_ts = None
        self.eid = sample_graph['eids']
        if mode == 'train':
            mask = sample_graph['train_mask']
        if mode == 'val':
            mask = sample_graph['val_mask']
        if mode == 'test':
            mask = sample_graph['test_mask']
        if mode != 'full':
            self.edge_index = self.edge_index[:, mask]
            self.edge_ts = self.edge_ts[mask]
            self.eid = self.eid[mask]  


        
    


