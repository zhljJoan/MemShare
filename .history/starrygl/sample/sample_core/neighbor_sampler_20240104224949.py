import starrygl
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__))))
import math
import torch
import torch.multiprocessing as mp
from typing import Optional, Tuple

from .base import BaseSampler, NegativeSampling, SampleOutput, SampleType
# from sample_cores import ParallelSampler, get_neighbors, heads_unique

from torch.distributed.rpc import rpc_async

class NeighborSampler(BaseSampler):
    """  Distributed Temporal Sampling:
    Parallel sampling is crucial for expanding model training to a large amount of data.
    Due to the large scale and complexity of graph data, traditional serial sampling may lead to significant
    waste of computing and storage resources. The significance of parallel sampling lies in improving the
    efficiency and overall computational speed of sampling by simultaneously sampling from multiple nodes or neighbors.
    This helps to accelerate the training and inference process of the model,
    making it more scalable and practical when dealing with large-scale graph data.
    
    Our parallel sampling adopts a hybrid approach of CPU and GPU, where the entire graph structure is stored on the    CPU
    and then uploaded to the GPU after sampling the graph structure on the CPU. Each trainer has a separate sampler     for parallel training.
    
    We have encapsulated the functions for parallel sampling, and you can easily use them in the following ways:
    
        .. code-block:: python
            # First,you need to import Python packages
            from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler
            # Then,you can use ours parallel sampler
            sampler = NeighborSampler(num_nodes=num_nodes, num_layers=num_layers, fanout=fanout, graph_data=graph_data,
            workers=workers, is_distinct = is_distinct, policy = policy, edge_weight= edge_weight, graph_name =     graph_name)
    
    Args:
        num_nodes: the num of all nodes in the graph
        num_layers: the num of layers to be sampled
        fanout: the list of max neighbors' number chosen for each layer
        graph_data: the graph data you want to sample
        workers: the number of threads, default value is 1
        is_distinct: 1-need distinct muti-edge, 0-don't need distinct muti-edge
        policy: "uniform" or "recent" or "weighted"
        edge_weight: the initial weights of edges
        graph_name: the name of graph
        should provide edge_index or (neighbors, deg)
    
    Examples:
    
        .. code-block:: python
    
            from starrygl.sample.part_utils.partition_tgnn import partition_load
            from starrygl.sample.graph_core import DataSet, DistributedGraphStore, TemporalNeighborSampleGraph
            from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler
    
            pdata = partition_load("PATH/{}".format(dataname), algo="metis_for_tgnn")
            graph = DistributedGraphStore(pdata = pdata,uvm_edge = False,uvm_node = False)
            sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
            sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=1, fanout=[10],graph_data=sample_graph,     workers=15,policy = 'recent',graph_name = "wiki_train")
    
    
    If you want to directly call parallel sampling functions, use the following methods:
    
        .. code-block:: python
    
            # the parameter meaning is the same as the `Args` above
            from starrygl.lib.libstarrygl_sampler import ParallelSampler, get_neighbors
    
            # get neighbor infomation table,row and col come from graph_data.edge_index=(row, col)
            tnb = get_neighbors(graph_name, row.contiguous(), col.contiguous(), num_nodes, is_distinct, graph_data. eid, edge_weight, timestamp)
    
            # call parallel sampler
            p_sampler = ParallelSampler(self.tnb, num_nodes, graph_data.num_edges, workers, fanout, num_layers, policy)
    
            For complete usage and more details, please refer to `starrygl.sample.sample_core.neighbor_sampler`"""
    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
        fanout: list,
        graph_data,
        workers = 1,
        tnb = None,
        is_distinct = 0,
        policy = "uniform",
        edge_weight: Optional[torch.Tensor] = None,
        graph_name = None
    ) -> None:
        r"""__init__
        Args:
            num_nodes: the num of all nodes in the graph
            num_layers: the num of layers to be sampled
            fanout: the list of max neighbors' number chosen for each layer
            workers: the number of threads, default value is 1
            tnb: neighbor infomation table
            is_distinct: 1-need distinct muti-edge, 0-don't need distinct muti-edge
            policy: "uniform" or "recent" or "weighted"
            edge_weight: the initial weights of edges
            graph_name: the name of graph
        should provide edge_index or (neighbors, deg)
        """
        super().__init__()
        self.num_layers = num_layers
        # 线程数不超过torch默认的omp线程数
        self.workers = workers # min(workers, torch.get_num_threads())
        self.fanout = fanout
        self.num_nodes = num_nodes
        self.graph_data=graph_data
        self.policy = policy
        self.is_distinct = is_distinct
        assert graph_name is not None
        self.graph_name = graph_name
        if(tnb is None):
            if(graph_data.edge_ts is not None):
                timestamp,ind = graph_data.edge_ts.sort()
                timestamp = timestamp.float().contiguous()
                eid = graph_data.eid[ind].contiguous()
                row, col = graph_data.edge_index[:,ind]
            else:
                eid = graph_data.eid
                timestamp = None
                row, col = graph_data.edge_index
            if(edge_weight is not None):
                edge_weight = edge_weight.float().contiguous()
            self.tnb = starrygl.sampler_ops.get_neighbors(graph_name, row.contiguous(), col.contiguous(), num_nodes, is_distinct, eid, edge_weight, timestamp)
        else:
            assert tnb is not None
            self.tnb = tnb
        self.p_sampler = starrygl.sampler_ops.ParallelSampler(self.tnb, num_nodes, graph_data.num_edges, workers, 
                                         fanout, num_layers, policy)
    
    def _get_sample_info(self):
        return self.num_nodes,self.num_layers,self.fanout,self.workers
    
    def _get_sample_options(self):
        return {"is_distinct" : self.is_distinct,
                "policy" : self.policy,
                "with_eid" : self.tnb.with_eid, 
                "weighted" : self.tnb.weighted, 
                "with_timestamp" : self.tnb.with_timestamp}
    
    def insert_edges_with_timestamp(
            self, 
            edge_index : torch.Tensor, 
            eid : torch.Tensor, 
            timestamp : torch.Tensor,
            edge_weight : Optional[torch.Tensor] = None):
        row, col = edge_index
        # 更新节点数和tnb
        self.num_nodes = self.tnb.update_neighbors_with_time(
            row.contiguous(), 
            col.contiguous(), 
            timestamp.contiguous(), 
            eid.contiguous(), 
            self.is_distinct, 
            edge_weight.contiguous())
    
    def update_edges_weight(
            self, 
            edge_index : torch.Tensor, 
            eid : torch.Tensor,
            edge_weight : Optional[torch.Tensor] = None):
        row, col = edge_index
        # 更新tnb的权重信息
        if self.tnb.with_eid:
            self.tnb.update_edge_weight(
                eid.contiguous(),
                col.contiguous(),
                edge_weight.contiguous()
            )
        else:
            self.tnb.update_edge_weight(
                row.contiguous(),
                col.contiguous(),
                edge_weight.contiguous()
            )
    
    def update_nodes_weight(
            self, 
            nid : torch.Tensor,
            node_weight : Optional[torch.Tensor] = None):
        # 更新tnb的权重信息
        self.tnb.update_node_weight(
            nid.contiguous(),
            node_weight.contiguous()
        )

    def update_all_node_weight(
            self,
            node_weight : torch.Tensor):
        # 更新tnb的权重信息
        self.tnb.update_all_node_weight(node_weight.contiguous())

    def sample_from_nodes(
        self,
        nodes: torch.Tensor,
        ts: Optional[torch.Tensor] = None,
        with_outer_sample: SampleType = SampleType.Whole
    ) -> SampleOutput:
        r"""Performs mutilayer sampling from the nodes specified in: nodes
        The specific number of layers is determined by parameter: num_layers
        returning a sampled subgraph in the specified output format: Tuple[torch.Tensor, list].

        Args:
            nodes: the list of seed nodes index,
            ts: the timestamp of nodes, optional,
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            fanout_index: optional. Specify the index to fanout
        Returns:
            sampled_nodes: the node sampled
            sampled_edge_index_list: the edge sampled
        """
        if(ts is None):
            self.part_unique = True
            self.p_sampler.neighbor_sample_from_nodes(nodes.contiguous(), None, self.part_unique)
            ret = self.p_sampler.get_ret()
            return ret
        else:
            self.p_sampler.neighbor_sample_from_nodes(nodes.contiguous(), ts.float().contiguous(), None)
            ret = self.p_sampler.get_ret()
            return ret
        
    def sample_from_edges(
        self,
        edges: torch.Tensor,
        ets: Optional[torch.Tensor] = None,
        neg_sampling: Optional[NegativeSampling] = None,
        with_outer_sample: SampleType = SampleType.Whole
    ) -> SampleOutput:
        r"""Performs sampling from the edges specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        Args:
            edges: the list of seed edges index
            with_outer_sample: 0-sample in whole graph structure; 1-sample onehop outer nodel; 2-cross partition sampling
            ets: the timestamp of edges, optional
            neg_sampling: The negative sampling configuration
        Returns:
            sampled_edge_index_list: the edges sampled
            sampled_eid_list: the edges' id sampled
            sampled_delta_ts_list:the edges' delta time sampled
            metadata: other infomation
        """
        src, dst = edges
        num_pos = src.numel()
        num_neg = 0
        with_timestap = ets is not None
        seed_ts = None

        if neg_sampling is not None:
            num_neg = math.ceil(num_pos * neg_sampling.amount)
            if neg_sampling.is_binary():
                src_neg = neg_sampling.sample(num_neg, self.num_nodes)
                dst_neg = neg_sampling.sample(num_neg, self.num_nodes)
                seed = torch.cat([src, dst, src_neg, dst_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets, ets], dim=0)
            if neg_sampling.is_triplet():
                src_neg = neg_sampling.sample(num_neg, self.num_nodes)
                seed = torch.cat([src, dst, src_neg], dim=0)
                if with_timestap: # ts操作
                    seed_ts = torch.cat([ets, ets, ets], dim=0)
        else:
            seed = torch.cat([src, dst], dim=0)            
            if with_timestap: # ts操作
                seed_ts = torch.cat([ets, ets], dim=0)

        # 去重负采样
        if neg_sampling is not None and neg_sampling.unique:
            if with_timestap: # ts操作
                pair, inverse_seed= torch.unique(torch.stack([seed, seed_ts],0), return_inverse=True, dim=1)
                seed, seed_ts = pair
                seed = seed.long()
            else:
                seed, inverse_seed = seed.unique(return_inverse=True)

        out = self.sample_from_nodes(seed, seed_ts, with_outer_sample)
        
        if neg_sampling is None or (not neg_sampling.unique):
            if with_timestap:
                return out, {'seed':seed,'seed_ts':seed_ts,
                             'src_pos_index':torch.arange(0,num_pos), 
                             'dst_pos_index':torch.arange(num_pos,2*num_pos), 
                             'src_neg_index':torch.arange(2*num_pos,3*num_pos)}
            else:
                return out, {'seed':seed,
                             'src_pos_index':slice(0,num_pos), 
                             'dst_pos_index':slice(num_pos,2*num_pos), 
                             'src_neg_index':slice(2*num_pos,3*num_pos)}

        metadata = {}
        if neg_sampling.is_binary():
            src_pos_index = inverse_seed[:num_pos]
            dst_pos_index = inverse_seed[num_pos:2 * num_pos]
            src_neg_index = inverse_seed[2 * num_pos:3 * num_pos]
            src_neg_index = src_neg_index.view(num_pos, -1).squeeze(-1)
            dst_neg_index = inverse_seed[3 * num_pos:]
            dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)
            metadata = {'seed':seed, 'src_neg_index':src_neg_index, 'dst_pos_index':dst_pos_index, 'dst_neg_index':dst_neg_index}
            if with_timestap:
                metadata['seed_ts'] = seed_ts
        elif neg_sampling.is_triplet():
            src_pos_index = inverse_seed[:num_pos]
            dst_pos_index = inverse_seed[num_pos:2 * num_pos]
            src_neg_index = inverse_seed[2 * num_pos:]
            src_neg_index = src_neg_index.view(num_pos, -1).squeeze(-1)
            # src_index是seed里src点的索引
            # dst_pos_index是seed里dst_pos点的索引
            # dst_neg_index是seed里dst_neg点的索引
            metadata = {'seed':seed, 'src_pos_index':src_pos_index, 'src_neg_index':src_neg_index, 'dst_pos_index':dst_pos_index}
            if with_timestap:
                metadata['seed_ts'] = seed_ts
        # sampled_nodes最前方是原始序列的采样起点也就是去重后的seed
        return out, metadata
    


if __name__=="__main__":
    # edge_index1 = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 5], # , 3, 3
    #                             [1, 0, 2, 4, 1, 3, 0, 3, 5, 0, 2]])# , 2, 5
    edge_index1 = torch.tensor([[0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 5], # , 3, 3
                                [1, 0, 2, 0, 4, 1, 3, 0, 3, 3, 5, 0, 2]])# , 2, 5
    edge_weight1 = None
    timeStamp=torch.FloatTensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
    num_nodes1 = 6
    num_neighbors = 2
    # Run the neighbor sampling
    from Utils import GraphData
    g_data = GraphData(id=0, edge_index=edge_index1, timestamp=timeStamp, data=None, partptr=torch.tensor([0, num_nodes1]))
    sampler = NeighborSampler(num_nodes=num_nodes1, 
                              num_layers=3, 
                              fanout=[2, 1, 1], 
                              edge_weight=edge_weight1, 
                              graph_data=g_data, 
                              graph_name='a',
                              workers=4,
                              is_distinct = 0)

    out = sampler.sample_from_nodes(torch.tensor([1,2]),
                                    ts=torch.tensor([1, 2]),
                                    with_outer_sample=SampleType.Whole)
    # out = sampler.sample_from_edges(torch.tensor([[1,2],[4,0]]), 
    #                                 with_outer_sample=SampleType.Whole, 
    #                                 ets = torch.tensor([1, 2]))
    
    # Print the result
    print('node:', out.node)
    print('edge_index_list:', out.edge_index_list)
    print('eid_list:', out.eid_list)
    print('delta_ts_list:', out.delta_ts_list)
    print('metadata: ', out.metadata)
