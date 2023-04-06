import torch
import torch.distributed as dist

from torch import Tensor, LongTensor
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.utils import degree

import os
import os.path as osp
import shutil

from typing import *



def partition_save(root: str, data: Data, num_parts: int, algo: str = "metis"):
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
    
    for i, pdata in enumerate(partition_data(data, num_parts, algo, verbose=True)):
        print(f"saving partition data: {i+1}/{num_parts}")
        fn = osp.join(path, f"{i:03d}")
        torch.save(pdata, fn)

def partition_load(root: str, algo: str = "metis") -> Data:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    fn = osp.join(root, f"{algo}_{world_size}", f"{rank:03d}")
    return torch.load(fn)

def partition_data(data: Data, num_parts: int, algo: str, verbose: bool = False) -> List[Data]:
    if algo == "metis":
        part_fn = metis_partition
    elif algo == "random":
        part_fn = random_partition
    else:
        raise ValueError(f"invalid algorithm: {algo}")
    
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    edge_index = data.edge_index
    
    if verbose: print(f"running partition algorithm: {algo}")
    node_parts, edge_parts = part_fn(edge_index, num_nodes, num_parts)
    
    if verbose: print("computing GCN normalized factor")
    gcn_norm = compute_gcn_norm(edge_index, num_nodes)
    
    if data.y.dtype == torch.long:
        if verbose: print("compute num_classes")
        num_classes = data.y.max().item() + 1
    else:
        num_classes = None

    for i in range(num_parts):
        npart_i = torch.where(node_parts == i)[0]
        epart_i = torch.where(edge_parts == i)[0]
        
        npart = npart_i
        epart = edge_index[:,epart_i]
        
        pdata = {
            "ids": npart,
            "edge_index": epart,
            "gcn_norm": gcn_norm[epart_i],
        }

        if num_classes is not None:
            pdata["num_classes"] = num_classes
    
        for key, val in data:
            if key == "edge_index":
                continue
            if isinstance(val, Tensor):
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

def compute_gcn_norm(edge_index: LongTensor, num_nodes: int) -> Tensor:
    deg_j = degree(edge_index[0], num_nodes).pow(-0.5)
    deg_i = degree(edge_index[1], num_nodes).pow(-0.5)
    deg_i[deg_i.isinf() | deg_i.isnan()] = 0.0
    deg_j[deg_j.isinf() | deg_j.isnan()] = 0.0
    return deg_j[edge_index[0]] * deg_i[edge_index[1]]

def _nopart(edge_index: LongTensor, num_nodes: int) -> Tuple[LongTensor, LongTensor]:
    node_parts = torch.zeros(num_nodes, dtype=torch.long)
    edge_parts = torch.zeros(edge_index.size(1), dtype=torch.long)
    return node_parts, edge_parts

def metis_partition(edge_index: LongTensor, num_nodes: int, num_parts: int) -> Tuple[LongTensor, LongTensor]:
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
    rowptr, col, _ = adj_t.csr()
    node_parts = torch.ops.torch_sparse.partition(rowptr, col, None, num_parts, num_parts < 8)
    edge_parts = node_parts[edge_index[1]]
    return node_parts, edge_parts


def random_partition(edge_index: LongTensor, num_nodes: int, num_parts: int) -> Tuple[LongTensor, LongTensor]:
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    
    node_parts = torch.randint(num_parts, size=(num_nodes,), dtype=edge_index.dtype)
    edge_parts = node_parts[edge_index[1]]
    return node_parts, edge_parts