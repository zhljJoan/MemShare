import torch

from torch import Tensor
from typing import *

import os
import os.path as osp
import shutil
from pathlib import Path
from torch_sparse import SparseTensor
from .partition import *

import logging

__all__ = [
    "GraphData",
    "partition_pyg",
    "partition_load",
]


class GraphData:
    def __init__(self,
        edge_indices: Union[Tensor, Dict[Tuple[str, str, str], Tensor]],
        num_nodes: Union[int, Dict[str, int]],
    ) -> None:
        if isinstance(edge_indices, Tensor):
            self._heterogeneous = False
            edge_indices = {("#", "@", "#"): edge_indices}
            num_nodes = {"#": int(num_nodes)}
        else:
            self._heterogeneous = True

        self._num_nodes: Dict[str, int] = {}
        self._node_data: Dict[str, 'NodeData'] = {}
        for ntype, num in num_nodes.items():
            ntype, num = str(ntype), int(num)
            self._num_nodes[ntype] = num
            self._node_data[ntype] = NodeData(ntype, num)
        
        self._edge_indices: Dict[Tuple[str, str, str], Tensor] = {}
        self._edge_data: Dict[Tuple[str, str, str], 'EdgeData'] = {}
        for (es, et, ed), edge_index in edge_indices.items():
            assert isinstance(edge_index, Tensor), f"edge_index must be a tensor, got {type(edge_index)}"
            assert edge_index.dim() == 2 and edge_index.size(0) == 2

            es, et, ed = str(es), str(et), str(ed)
            assert es in self._num_nodes, f"unknown node type '{es}', should be one of {list(self._num_nodes.keys())}."
            assert ed in self._num_nodes, f"unknown node type '{ed}', should be one of {list(self._num_nodes.keys())}."

            etype = (es, et, ed)
            self._edge_indices[etype] = edge_index
            self._edge_data[etype] = EdgeData(etype, edge_index.size(1))
        
        self._meta = MetaData()
    
    def meta(self) -> 'MetaData':
        return self._meta

    def node(self, node_type: Optional[str] = None) -> 'NodeData':
        if len(self._node_data) == 1:
            for data in self._node_data.values():
                return data
        return self._node_data[node_type]
    
    def edge(self, edge_type: Optional[Tuple[str, str, str]] = None) -> 'EdgeData':
        if len(self._edge_data) == 1:
            for data in self._edge_data.values():
                return data
        return self._edge_data[edge_type]
    
    def edge_index(self, edge_type: Optional[Tuple[str, str, str]] = None) -> Tensor:
        if len(self._edge_indices) == 1:
            for data in self._edge_indices.values():
                return data
        return self._edge_indices[edge_type]
    
    @property
    def is_heterogeneous(self) -> bool:
        return self._heterogeneous
    
    def to(self, device: Any) -> 'GraphData':
        self._meta.to(device)
        
        for ndata in self._node_data.values():
            ndata.to(device)
        
        for edata in self._edge_data.values():
            edata.to(device)

        self._edge_indices = {k:v.to(device) for k,v in self._edge_indices.items()}

        return self

class MetaData:
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def keys(self) -> List[str]:
        return list(self._data.keys())
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __setitem__(self, key: str, val: Any):
        assert isinstance(key, str)
        self._data[key] = val
    
    def pop(self, key: str) -> Tensor:
        if key in self._data:
            return self._data.pop(key)
    
    def to(self, device: Any) -> 'MetaData':
        for k in self.keys():
            v = self._data[k]
            if isinstance(v, Tensor):
                self._data[k] = v.to(device)
        return self

class NodeData:
    def __init__(self,
        node_type: str,
        num_nodes: int,
    ) -> None:
        self._node_type = str(node_type)
        self._num_nodes = int(num_nodes)
        self._data: Dict[str, Tensor] = {}
    
    @property
    def node_type(self) -> str:
        return self._node_type
    
    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    def keys(self) -> List[str]:
        return list(self._data.keys())
    
    def __getitem__(self, key: str) -> Tensor:
        return self._data[key]
    
    def __setitem__(self, key: str, val: Tensor):
        assert isinstance(key, str)
        assert val.size(0) == self._num_nodes
        self._data[key] = val
    
    def pop(self, key: str) -> Tensor:
        if key in self._data:
            return self._data.pop(key)
    
    def to(self, device: Any) -> 'NodeData':
        self._data = {k:v.to(device) for k,v in self._data.items()}
        return self


class EdgeData:
    def __init__(self,
        edge_type: Tuple[str, str, str],
        num_edges: int,
    ) -> None:
        self._edge_type = tuple(str(t) for t in edge_type)
        self._num_edges = num_edges
        assert len(self._edge_type) == 3

        self._data: Dict[str, Tensor] = {}
    
    @property
    def edge_type(self) -> Tuple[str, str, str]:
        return self._edge_type
    
    @property
    def num_edges(self) -> int:
        return self._num_edges
    
    def keys(self) -> List[str]:
        return list(self._data.keys())
    
    def __getitem__(self, key: str) -> Tensor:
        return self._data[key]
    
    def __setitem__(self, key: str, val: Optional[Tensor]) -> Tensor:
        assert isinstance(key, str)
        assert val.size(0) == self._num_edges
        self._data[key] = val
    
    def pop(self, key: str) -> Tensor:
        if key in self._data:
            return self._data.pop(key)

    def to(self, device: Any) -> 'EdgeData':
        self._data = {k:v.to(device) for k,v in self._data.items()}
        return self


def partition_load(root: str, part_id: int, num_parts: int, algo: str = "metis") -> GraphData:
    p = Path(root).expanduser().resolve() / f"{algo}_{num_parts}" / f"{part_id:03d}"
    return torch.load(p.__str__())


def partition_pyg(root: str, data, num_parts: int, algo: str = "metis"):
    root_path = Path(root).expanduser().resolve()
    base_path = root_path / f"{algo}_{num_parts}"

    if base_path.exists():
        shutil.rmtree(base_path.__str__())
    base_path.mkdir(parents=True, exist_ok=True)
    
    for i, g in enumerate(partition_pyg_data(data, num_parts, algo)):
        logging.info(f"saving partition data: {i+1}/{num_parts}")
        torch.save(g, (base_path / f"{i:03d}").__str__())


def partition_pyg_data(data, num_parts: int, algo: str = "metis") -> Iterator[GraphData]:
    from torch_geometric.data import Data
    assert isinstance(data, Data), f"must be Data class in pyg"
    
    logging.info(f"running partition aglorithm: {algo}")
    
    num_nodes: int = data.num_nodes
    num_edges: int = data.num_edges
    edge_index: Tensor = data.edge_index

    if algo == "metis":
        node_parts = metis_partition(edge_index, num_nodes, num_parts)
    elif algo == "mt-metis":
        node_parts = mt_metis_partition(edge_index, num_nodes, num_parts)
    elif algo == "random":
        node_parts = random_partition(edge_index, num_nodes, num_parts)
    else:
        raise ValueError(f"unknown partition algorithm: {algo}")
    
    if data.y.dtype == torch.long:
        if data.y.dim() == 1:
            num_classes = data.y.max().item() + 1
        else:
            num_classes = data.y.size(1)
    else:
        num_classes = None

    for i in range(num_parts):
        npart_mask = node_parts == i
        epart_mask = npart_mask[edge_index[1]]

        local_edges = edge_index[:, epart_mask]

        raw_src_ids: Tensor = local_edges[0].unique()
        raw_dst_ids: Tensor = torch.where(npart_mask)[0]
        
        M: int = raw_src_ids.max().item() + 1
        imap = torch.full((M,), (2**62-1)*2+1).type_as(raw_src_ids)
        imap[raw_src_ids] = torch.arange(raw_src_ids.numel()).type_as(raw_src_ids)
        local_src = imap[local_edges[0]]
        
        M: int = raw_dst_ids.max().item() + 1
        imap = torch.full((M,), (2**62-1)*2+1).type_as(raw_dst_ids)
        imap[raw_dst_ids] = torch.arange(raw_dst_ids.numel()).type_as(raw_dst_ids)
        local_dst = imap[local_edges[1]]

        local_edges = torch.vstack([local_src, local_dst])

        g = GraphData(
            edge_indices={
                ("src", "@", "dst"): local_edges,
            },
            num_nodes={
                "src": raw_src_ids.numel(),
                "dst": raw_dst_ids.numel(),
            },
        )
        g.node("src")["raw_ids"] = raw_src_ids
        g.node("dst")["raw_ids"] = raw_dst_ids

        if num_classes is not None:
            g.meta()["num_classes"] = num_classes

        for key, val in data:
            if key == "edge_index":
                continue
            elif isinstance(val, Tensor):
                if val.size(0) == num_nodes:
                    g.node("dst")[key] = val[npart_mask]
                elif val.size(0) == num_edges:
                    g.edge()[key] = val[epart_mask]
            elif isinstance(val, SparseTensor):
                pass
            else:
                g.meta()[key] = val
        yield g
