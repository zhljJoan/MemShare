from typing import Optional
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor
from starrygl.sample.cache import LRU_cache
from starrygl.sample.cache.cache import Cache
from starrygl.sample.cache.static_cache import StaticCache
from starrygl.sample.cache.utils import pre_sample
from starrygl.sample.graph_core import DistributedGraphStore
from starrygl.sample.memory.shared_mailbox import SharedMailBox
import torch

_FetchCache = None        
class FetchFeatureCache:
    @staticmethod
    def create_fetch_cache(num_nodes: int, num_edges: int,
                 edge_cache_ratio: int, node_cache_ratio: int,
                 graph: DistributedGraphStore,
                 mailbox:SharedMailBox = None,
                 policy = 'lru'):
        """
        method to create a fetch cache instance.

        Args:
            num_nodes: Total number of nodes in the graph.
            num_edges: Total number of edges in the graph.
            edge_cache_ratio: The hit rate of cache edges.
            node_cache_ratio: The hit rate of cache nodes.
            graph: Distributed graph store.
            mailbox: used for storing information.
            policy: Caching policy, either 'lru' or 'static'.

        """
        global _FetchCache
        _FetchCache = FetchFeatureCache(num_nodes, num_edges,
                 edge_cache_ratio, node_cache_ratio,
                 graph,mailbox,policy)
    @staticmethod
    def getFetchCache():
        """
        method to get the existing fetch cache instance.

        Returns:
            FetchFeatureCache: The existing fetch cache instance.
        """
        global _FetchCache
        return _FetchCache
    def __init__(self, num_nodes: int, num_edges: int,
                 edge_cache_ratio: int, node_cache_ratio: int,
                 graph: DistributedGraphStore,
                 mailbox:SharedMailBox = None,
                 policy = 'lru'
                ):
        """
        Initializes the FetchFeatureCache instance.

        Args:
            num_nodes: Total number of nodes in the graph.
            num_edges: Total number of edges in the graph.
            edge_cache_ratio: The hit rate of cache edges.
            node_cache_ratio: The hit rate of cache nodes.
            graph: Distributed graph store.
            mailbox: used for storing information.
            policy: Caching policy, either 'lru' or 'static'.

        """
        if policy == 'lru':
            init_fn = LRU_cache.LRUCache
        elif policy == 'static':
            init_fn = StaticCache
        self.ctx = DistributedContext.get_default_context()
        if graph.x is not None:
            self.node_cache:Cache = init_fn(node_cache_ratio,num_nodes,
                                    [graph.x],use_local=graph.uvm_node)
        else:
            self.node_cache = None
        if graph.edge_attr is not None:
            self.edge_cache:Cache = init_fn(edge_cache_ratio,num_edges,
                                    [graph.edge_attr],use_local = graph.uvm_edge)
        else:
            self.edge_cache = None
        if mailbox is not None:
            self.mailbox_cache:Cache = init_fn(node_cache_ratio,num_nodes,
                                       [mailbox.node_memory,
                                        mailbox.node_memory_ts.accessor.data.reshape(-1,1),
                                        mailbox.mailbox,
                                        mailbox.mailbox_ts],
                                       use_local = mailbox.uvm)
        else:
            self.mailbox_cache = None
        self.graph = graph
        self.mailbox = mailbox
        global FetchCache 
        FetchCache = self    

    def fetch_feature(self, nid: Optional[torch.Tensor] = None, dist_nid = None,
                      eid: Optional[torch.Tensor] = None,  dist_eid = None
                      ):
        """
        Fetches node and edge features along with mailbox memory.

        Args:
            nid: Node indices to fetch features for.
            dist_nid: The remote communication corresponding to nid.
            eid: Edge indices to fetch features for.
            dist_eid: The remote communication corresponding to eid.

        """
        nfeat = None
        mem = None
        efeat = None
        if self.node_cache is not None and nid is not  None:
            nfeat = torch.zeros(nid.shape[0],
                                self.node_cache.buffers[0].shape[1],
                                dtype = self.node_cache.buffers[0].dtype,
                                device = torch.device('cuda')
                                )
            if self.node_cache.use_local is False:
                local_mask =  (DistIndex(dist_nid).part == torch.distributed.get_rank())
                local_id = dist_nid[local_mask]
                nfeat[local_mask] = self.graph.x.accessor.data[DistIndex(local_id).loc]
                remote_mask = ~local_mask
                if remote_mask.sum() > 0:
                    remote_id = nid[remote_mask]
                    source_id = dist_nid[remote_mask]
                    nfeat[remote_mask] = self.node_cache.fetch_data(remote_id,\
                                            self.graph._get_node_attr,source_id)[0]
            else:
                nfeat = self.node_cache.fetch_data(nid,
                                        self.graph._get_node_attr,dist_nid)[0]
        if self.mailbox_cache is not None and nid is not  None:
            memory = torch.zeros(nid.shape[0],
                                self.mailbox_cache.buffers[0].shape[1],
                                dtype = self.mailbox_cache.buffers[0].dtype,
                                device = torch.device('cuda')
                            )
            memory_ts = torch.zeros(nid.shape[0],
                                dtype = self.mailbox_cache.buffers[1].dtype,
                                device = torch.device('cuda')
                            )
            mailbox = torch.zeros(nid.shape[0],
                                *self.mailbox_cache.buffers[2].shape[1:],
                                dtype = self.mailbox_cache.buffers[2].dtype,
                                device = torch.device('cuda')
                                )
            mailbox_ts = torch.zeros(nid.shape[0],
                                *self.mailbox_cache.buffers[3].shape[1:],
                                dtype = self.mailbox_cache.buffers[3].dtype,
                                device = torch.device('cuda')
                                )
            if self.mailbox_cache.use_local is False:
                if self.node_cache is None:
                    local_mask =  (DistIndex(dist_nid).part == torch.distributed.get_rank())
                    local_id = dist_nid[local_mask]
                    remote_mask = ~local_mask
                    remote_id = nid[remote_mask]
                    source_id = dist_nid[remote_mask]
                mem = self.mailbox.gather_memory(local_id)
                memory[local_mask],memory_ts[local_mask],mailbox[local_mask],mailbox_ts[local_mask]= mem
                if remote_mask.sum() > 0:
                    mem = self.mailbox_cache.fetch_data(remote_id,\
                                        self.mailbox.gather_memory,source_id)
                    memory[remote_mask] = mem[0]
                    memory_ts[remote_mask] = mem[1].reshape(-1)
                    mailbox[remote_mask] = mem[2]
                    mailbox_ts[remote_mask] = mem[3]
                mem = memory,memory_ts,mailbox,mailbox_ts
            else:
                mem = self.mailbox_cache.fetch_data(nid,mailbox.gather_memory,dist_nid)
        if self.edge_cache is not None and eid is not None:
            efeat = torch.zeros(eid.shape[0],
                                self.edge_cache.buffers[0].shape[1],
                                dtype = self.edge_cache.buffers[0].dtype,
                                device = torch.device('cuda')
                                )
            if self.edge_cache.use_local is False:
                local_mask =  (DistIndex(dist_eid).part == torch.distributed.get_rank())
                local_id = dist_eid[local_mask]
                efeat[local_mask] = self.graph.edge_attr.accessor.data[DistIndex(local_id).loc]
                remote_mask = ~local_mask
                if remote_mask.sum() > 0:
                    remote_id = eid[remote_mask]
                    source_id = dist_eid[remote_mask]
                    efeat[remote_mask] = self.edge_cache.fetch_data(remote_id,\
                                        self.graph._get_edge_attr,source_id)[0]
            else:
                efeat = self.node_cache.fetch_data(eid,
                                        self.graph._get_edge_attr,dist_eid)[0]
        
        return nfeat,efeat,mem

    def init_cache_with_presample(self,dataloader, num_epoch:int = 10):
        """
        Initializes the cache with pre-sampled data from the provided dataloader.

        Args:
            dataloader: The data loader we implement, containing the graph data.
            num_epoch: Number of epochs to pre-sample the data.

        """
        node_size = self.node_cache.capacity if self.node_cache is not None else 0
        edge_size = self.edge_cache.capacity if self.edge_cache is not None else 0
        node_counts,edge_counts = pre_sample(dataloader=dataloader,
                             num_epoch=num_epoch,
                             node_size = node_size,
                             edge_size = edge_size)
        if node_size != 0:
            if self.node_cache.use_local is False:
                dist_mask = DistIndex(self.graph.nids_mapper).part == torch.distributed.get_rank()
                dist_mask = ~dist_mask
                node_counts = node_counts[dist_mask]
            _,nid = node_counts.topk(node_size)
            if self.node_cache.use_local is False:
                nid = dist_mask.nonzero()[nid]
            dist_nid = self.graph.nids_mapper[nid].unique()
            node_feature = self.graph._get_node_attr(dist_nid.to(self.graph.x.device))
            _nid = nid.reshape(-1)
            self.node_cache.init_cache(_nid,node_feature)
        print('finish node init')
        if edge_size != 0:
            if self.edge_cache.use_local is False:
                dist_mask = DistIndex(self.graph.eids_mapper).part == torch.distributed.get_rank()
                dist_mask = ~dist_mask
                edge_counts = edge_counts[dist_mask]
            _,eid = edge_counts.topk(edge_size)
            if self.edge_cache.use_local is False:
                eid_ = dist_mask.nonzero()[eid]
            else:
                eid_ = eid
            dist_eid = self.graph.eids_mapper[eid_].unique()
            edge_feature = self.graph._get_edge_attr(dist_eid.to(self.graph.edge_attr.device))
            eid_ = eid_.reshape(-1)
            self.edge_cache.init_cache(eid_,edge_feature)
        print('finish edge init')