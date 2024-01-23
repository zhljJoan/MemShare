from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import degree
import os.path as osp
import os
import shutil
import torch
import torch.utils.data

import torch.distributed as dist
from starrygl.lib.libstarrygl_sampler import get_norm_temporal
from starrygl.utils.partition import mt_metis_partition


def partition_load(root: str, algo: str = "metis") -> Data:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    fn = osp.join(root, f"{algo}_{world_size}", f"{rank:03d}")
    return torch.load(fn)


def partition_save(root: str, data: Data, num_parts: int,
                   algo: str = "metis",
                   node_weight = None,
                   edge_weight_dict=None):
    root = osp.abspath(root)
    if osp.exists(root) and not osp.isdir(root):
        raise ValueError(f"path '{root}' should be a directory")
    
    path = osp.join(root, f"{algo}_{num_parts}")
    if osp.exists(path) and not osp.isdir(path):
        raise ValueError(f"path '{path}' should be a directory")
    
    if osp.exists(path) and os.listdir(path):
        print(f"directory '{path}' not empty and cleared")
        for p in os.listdir(path):
            p = osp.join(path, p)
            if osp.isdir(p):
                shutil.rmtree(osp.join(path, p))
            else:
                os.remove(p)
                
    if not osp.exists(path):
        print(f"creating directory '{path}'")
        os.makedirs(path)
    if algo == 'metis_for_tgnn':
        for i, pdata in enumerate(partition_data_for_tgnn(
                data, num_parts, algo, verbose=True,
                node_weight = node_weight,
                edge_weight_dict=edge_weight_dict)):
            print(f"saving partition data: {i+1}/{num_parts}")
            fn = osp.join(path, f"{i:03d}")
            torch.save(pdata, fn)
    else:
        for i, pdata in enumerate(partition_data_for_gnn(data, num_parts,
                                                         algo, verbose=True)):
            print(f"saving partition data: {i+1}/{num_parts}")
            fn = osp.join(path, f"{i:03d}")
            torch.save(pdata, fn)


def partition_data_for_gnn(data: Data, num_parts: int, 
                           algo: str, verbose: bool = False):
    if algo == "metis":
        part_fn = metis_partition

    else:
        raise ValueError(f"invalid algorithm: {algo}")
    
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    edge_index = data.edge_index
    
    if verbose:
        print(f"running partition algorithm: {algo}")
    node_parts, edge_parts = part_fn(edge_index, num_nodes, num_parts)
    
    if verbose:
        print("computing GCN normalized factor")
    gcn_norm = compute_gcn_norm(edge_index, num_nodes)
    
    if data.y.dtype == torch.long:
        if verbose:
            print("compute num_classes")
        num_classes = data.y.max().item() + 1
    else:
        num_classes = None
        eids = torch.zeros(num_edges, dtype=torch.long)
    len = 0
    edgeptr = torch.zeros(num_parts+1, dtype=eids.dtype)
    for i in range(num_parts):
        epart_i = torch.where(edge_parts == i)[0]
        eids[epart_i] = torch.arange(epart_i.shape[0]) + len
        len += epart_i.shape[0]
        edgeptr[i+1] = len
    data.eids = eids
    data.sample_graph.sample_eids = eids[data.sample_graph.sample_eid]
    nids = torch.zeros(num_nodes, dtype=torch.long)
    len = 0
    partptr = torch.zeros(num_parts+1, dtype=nids.dtype)
    for i in range(num_parts):
        npart_i = torch.where(node_parts == i)[0]
        nids[npart_i] = torch.arange(npart_i.shape[0]) + len
        len += npart_i.shape[0]
        partptr[i+1] = len
    data.edge_index = nids[data.edge_index]
    data.sample_graph.edge_index = nids[data.sample_graph.edge_index]
    for i in range(num_parts):
        npart_i = torch.where(node_parts == i)[0]
        epart_i = torch.where(edge_parts == i)[0]
        
        npart = npart_i
        epart = edge_index[:, epart_i]
        
        pdata = {
            "ids": npart,
            "edge_index": epart,
            "gcn_norm": gcn_norm[epart_i],
            "sample_graph": data.sample_graph,
            "partptr": partptr,
            "edgeptr": edgeptr
        }

        if num_classes is not None:
            pdata["num_classes"] = num_classes
    
        for key, val in data:   
            if key == "edge_index" or key == "sample_graph":
                continue
            if isinstance(val, torch.Tensor):
                if val.size(0) == num_nodes:
                    pdata[key] = val[npart_i]
                elif val.size(0) == num_edges:
                    pdata[key] = val[epart_i]
                # else:
                #     pdata[key] = val
            elif isinstance(val, SparseTensor):
                pass
            else:
                pdata[key] = val
        
        pdata = Data(**pdata)
        yield pdata


def _nopart(edge_index: torch.LongTensor, num_nodes: int):
    node_parts = torch.zeros(num_nodes, dtype=torch.long)
    if isinstance(edge_index, torch.Tensor):
        edge_parts = torch.zeros(edge_index.size(1), dtype=torch.long)
        return node_parts, edge_parts
    return node_parts


def metis_for_tgnn(edge_index_dict: dict,
                   num_nodes: int,
                   num_parts: int,
                   node_weight = None,
                   edge_weight_dict=None):
    if num_parts <= 1:
        return _nopart(edge_index_dict, num_nodes)
    edge_list = []
    weight_list = []
    for i,key in enumerate(edge_index_dict):
        v = edge_index_dict[key]
        edge_list.append(v)
        weight_list.append(torch.ones(v.shape[1])*edge_weight_dict[key])
    edge_index = torch.cat(edge_list,dim = 1)
    edge_weight = torch.cat(weight_list,dim = 0)
    node_parts = mt_metis_partition(edge_index,num_nodes,num_parts,node_weight,edge_weight)
    return node_parts
    #G = nx.Graph()
    #G.add_nodes_from(torch.arange(0, num_nodes).tolist())
    #value, counts = torch.unique(edge_index_dict['edata'][1, :].view(-1),
    #                             return_counts=True)
    #nodes = torch.tensor(list(G.adj.keys()))
    #for i in range(value.shape[0]):
    #    if (value[i].item() in G.nodes):
    #        G.nodes[int(value[i].item())]['weight'] = counts[i]
    #        G.nodes[int(value[i].item())]['ones'] = 1
    #G.graph['node_weight_attr'] = ['weight', 'ones']
    #for i, key in enumerate(edge_index_dict):
    #    v = edge_index_dict[key]
    #    edges = torch.cat((v, (torch.ones(v.shape[1], dtype=torch.long) *
    #                           edge_weight_dict[key]).unsqueeze(0)), dim=0)
    #    # w = edges.T
    #    G.add_weighted_edges_from((edges.T).tolist())
    #G.graph['edge_weight_attr'] = 'weight'
    #cuts, part = metis.part_graph(G, num_parts)
    #node_parts = torch.zeros(num_nodes, dtype=torch.long)
    #node_parts[nodes] = torch.tensor(part)
    #return node_parts


"""
weight: 各种工作负载边划分权重
按照点均衡划分
"""


def partition_data_for_tgnn(data: Data, num_parts: int, algo: str,
                            verbose: bool = False,
                            node_weight: torch.Tensor = None,
                            edge_weight_dict: dict = None):
    if algo == "metis_for_tgnn":
        part_fn = metis_for_tgnn

    else:
        raise ValueError(f"invalid algorithm: {algo}")
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    edge_index_dict = data.edge_index_dict
    tgnn_norm = compute_temporal_norm(data.edge_index, data.edge_ts, num_nodes)
    if verbose:
        print(f"running partition algorithm: {algo}")
    node_parts = part_fn(edge_index_dict, num_nodes, num_parts,
                         node_weight,
                         edge_weight_dict)
    edge_parts = node_parts[data.edge_index[1, :]]
    eids = torch.arange(num_edges, dtype=torch.long)
    data.eids = eids
    data.sample_graph['eids'] = eids[data.sample_graph['eids']]
    if data.y.dtype == torch.long:
        if verbose:
            print("compute num_classes")
        num_classes = data.y.max().item() + 1
    else:
        num_classes = None

    for i in range(num_parts):
        npart_i = torch.where(node_parts == i)[0]        
        epart_i = torch.where(edge_parts == i)[0]

        pdata = {
            "ids": npart_i,
            "tgnn_norm": tgnn_norm,
            "edge_index": data.edge_index[:, epart_i],
            "sample_graph": data.sample_graph
        } 
        if num_classes is not None:
            pdata["num_classes"] = num_classes
    
        for key, val in data:
            if key == "edge_index" or key == "edge_index_dict" \
                    or key == "sample_graph":
                continue
            if isinstance(val, torch.Tensor):
                if val.size(0) == num_nodes:
                    pdata[key] = val[npart_i]
                elif val.size(0) == num_edges:
                    pdata[key] = val[epart_i]
                # else:
                #     pdata[key] = val
            elif isinstance(val, SparseTensor):
                pass
            else:
                pdata[key] = val      
        pdata = Data(**pdata)
        yield pdata


def metis_partition(edge_index, num_nodes: int, num_parts: int):
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    G = nx.Graph()
    G.add_nodes_from(torch.arange(0, num_nodes).tolist())
    G.add_edges_from(edge_index.T.tolist())
    nodes = torch.tensor(list(G.adj.keys()))
    nodes = torch.tensor(list(G.adj.keys()))
    cuts, part = metis.part_graph(G, num_parts)
    node_parts = torch.zeros(num_nodes, dtype=torch.long)
    node_parts[nodes] = torch.tensor(part)
    edge_parts = node_parts[edge_index[1]]
    return node_parts, edge_parts


def metis_partition_bydegree(edge_index, num_nodes: int, num_parts: int):
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    G = nx.Graph()
    G.add_nodes_from(torch.arange(0, num_nodes).tolist())
    G.add_edges_from(edge_index.T.tolist())
    value, counts = torch.unique(edge_index[1, :].view(-1), return_counts=True)
    nodes = torch.tensor(list(G.adj.keys()))
    for i in range(value.shape[0]):
        if (value[i].item() in G.nodes):
            G.nodes[int(value[i].item())]['weight'] = counts[i]
    G.graph['node_weight_attr'] = 'weight'
    nodes = torch.tensor(list(G.adj.keys()))
    cuts, part = metis.part_graph(G, num_parts)
    node_parts = torch.zeros(num_nodes, dtype=torch.long)
    node_parts[nodes] = torch.tensor(part)
    edge_parts = node_parts[edge_index[1]]
    return node_parts, edge_parts


def compute_gcn_norm(edge_index: torch.LongTensor, num_nodes: int):
    deg_j = degree(edge_index[0], num_nodes).pow(-0.5)
    deg_i = degree(edge_index[1], num_nodes).pow(-0.5)
    deg_i[deg_i.isinf() | deg_i.isnan()] = 0.0
    deg_j[deg_j.isinf() | deg_j.isnan()] = 0.0
    return deg_j[edge_index[0]] * deg_i[edge_index[1]]


def compute_temporal_norm(edge_index: torch.LongTensor,
                          timestamp: torch.FloatTensor,
                          num_nodes: int):
    srcavg, srcvar, dstavg, dstvar = get_norm_temporal(edge_index[0, :],
                                                       edge_index[1, :],
                                                       timestamp, num_nodes)
    return srcavg, srcvar, dstavg, dstvar

