import torch
import starrygl

from torch import Tensor
from torch_sparse import SparseTensor

from typing import *


__all__ = [
    "metis_partition",
    "mt_metis_partition",
    "random_partition",
    "pyg_metis_partition"
]

def _nopart(edge_index: Tensor, num_nodes: int):
    return torch.zeros(num_nodes).type_as(edge_index)

def metis_partition(
    edge_index: Tensor,
    num_nodes: int,
    num_parts: int,
    node_weight: Optional[Tensor] = None,
    edge_weight: Optional[Tensor] = None,
    node_sizes: Optional[Tensor] = None,
    recursive: bool = False,
    min_edge_cut: bool = False,
) -> Tensor:
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    
    adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes))
    rowptr, col, value = adj_t.coalesce().to_symmetric().csr()
    node_parts = starrygl.ops.metis_partition(
        rowptr, col, value, node_weight, node_sizes, num_parts, recursive, min_edge_cut)
    return node_parts

def pyg_metis_partition(
    edge_index: Tensor,
    num_nodes: int,
    num_parts: int,
) -> Tensor:
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
    rowptr, col, _ = adj_t.coalesce().to_symmetric().csr()
    node_parts = torch.ops.torch_sparse.partition(rowptr, col, None, num_parts, num_parts < 8)
    return node_parts

def mt_metis_partition(
    edge_index: Tensor,
    num_nodes: int,
    num_parts: int,
    node_weight: Optional[Tensor] = None,
    edge_weight: Optional[Tensor] = None,
    num_workers: int = 8,
    recursive: bool = False,
) -> Tensor:
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    
    adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes))
    rowptr, col, value = adj_t.coalesce().to_symmetric().csr()
    node_parts = starrygl.ops.mt_metis_partition(
        rowptr, col, value, node_weight, num_parts, num_workers, recursive)
    return node_parts


def random_partition(edge_index: Tensor, num_nodes: int, num_parts: int) -> Tensor:
    if num_parts <= 1:
        return _nopart(edge_index, num_nodes)
    return torch.randint(num_parts, size=(num_nodes,)).type_as(edge_index)
