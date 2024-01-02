import torch

from torch import Tensor
from typing import *

import shutil
from pathlib import Path

from torch_sparse import SparseTensor
from starrygl.utils.partition import *
from starrygl.parallel.route import Route
from starrygl.parallel.sparse import SparseBlocks

from .utils import init_vc_edge_index

import logging

__all__ = [
    "GraphData",
]


Strings = Sequence[str]
OptStrings = Optional[Strings]

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
    
    def node_types(self) -> List[str]:
        return list(self._node_data.keys())
    
    def edge_types(self) -> List[Tuple[str, str, str]]:
        return list(self._edge_data.keys())
    
    def to_route(self, group: Any = None) -> Route:
        src_ids = self.node("src")["raw_ids"]
        dst_ids = self.node("dst")["raw_ids"]
        return Route.from_raw_indices(src_ids, dst_ids, group=group)
    
    def to_sparse(self, key: Optional[str] = None, group: Any = None) -> SparseBlocks:
        src_ids = self.node("src")["raw_ids"]
        dst_ids = self.node("dst")["raw_ids"]
        edge_index = self.edge_index()
        edge_index = torch.vstack([
            src_ids[edge_index[0]],
            dst_ids[edge_index[1]],
        ])
        edge_attr = None if key is None else self.edge()[key]
        return SparseBlocks.from_raw_indices(dst_ids, edge_index, edge_attr=edge_attr, group=group)
    
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
    
    @staticmethod
    def from_bipartite(
        edge_index: Tensor,
        num_src_nodes: Optional[int] = None,
        num_dst_nodes: Optional[int] = None,
        raw_src_ids: Optional[Tensor] = None,
        raw_dst_ids: Optional[Tensor] = None,
    ) -> 'GraphData':
        if num_src_nodes is None:
            num_src_nodes = raw_src_ids.numel()
        
        if num_dst_nodes is None:
            num_dst_nodes = raw_dst_ids.numel()

        g = GraphData(
            edge_indices={
                ("src", "@", "dst"): edge_index,
            },
            num_nodes={
                "src": num_src_nodes,
                "dst": num_dst_nodes,
            }
        )

        if raw_src_ids is not None:
            g.node("src")["raw_ids"] = raw_src_ids
        
        if raw_dst_ids is not None:
            g.node("dst")["raw_ids"] = raw_dst_ids
        
        return g
    
    @staticmethod
    def from_pyg_data(data) -> 'GraphData':
        from torch_geometric.data import Data
        assert isinstance(data, Data), f"must be Data class in pyg"

        g = GraphData(data.edge_index, data.num_nodes)
        for key, val in data:
            if key == "edge_index":
                continue
            elif isinstance(val, Tensor):
                if val.size(0) == data.num_nodes:
                    g.node()[key] = val
                elif val.size(0) == data.num_edges:
                    g.edge()[key] = val
            elif isinstance(val, SparseTensor):
                logging.warning(f"found sparse matrix {key}, but ignored.")
            else:
                g.meta()[key] = val
        return g
    
    @staticmethod
    def load_partition(
        root: str,
        part_id: int,
        num_parts: int,
        algorithm: str = "metis",
    ) -> 'GraphData':
        p = Path(root).expanduser().resolve() / f"{algorithm}_{num_parts}" / f"{part_id:03d}"
        return torch.load(p.__str__())
    
    def save_partition(self,
        root: str,
        num_parts: int,
        node_weight: Optional[str] = None,
        edge_weight: Optional[str] = None,
        include_node_attrs: Optional[Sequence[str]] = None,
        include_edge_attrs: Optional[Sequence[str]] = None,
        include_meta_attrs: Optional[Sequence[str]] = None,
        ignore_node_attrs: Optional[Sequence[str]] = None,
        ignore_edge_attrs: Optional[Sequence[str]] = None,
        ignore_meta_attrs: Optional[Sequence[str]] = None,
        algorithm: str = "metis",
        partition_kwargs = None,
    ):
        assert not self.is_heterogeneous, "only support homomorphic graph"
    
        num_nodes: int = self.node().num_nodes
        edge_index: Tensor = self.edge_index()

        logging.info(f"running partition aglorithm: {algorithm}")
        partition_kwargs = partition_kwargs or {}
        
        not_self_loop = (edge_index[0] != edge_index[1])

        if node_weight is not None:
            node_weight = self.node()[node_weight]

        if edge_weight is not None:
            edge_weight = self.edge()[edge_weight]
            edge_weight = edge_weight[not_self_loop]
        
        if algorithm == "metis":
            node_parts = metis_partition(
                edge_index[:,not_self_loop],
                num_nodes, num_parts,
                node_weight=node_weight,
                edge_weight=edge_weight,
                **partition_kwargs,
            )
        elif algorithm == "mt-metis":
            node_parts = mt_metis_partition(
                edge_index[:,not_self_loop],
                num_nodes, num_parts,
                node_weight=node_weight,
                edge_weight=edge_weight,
                **partition_kwargs,
            )
        elif algorithm == "random":
            node_parts = random_partition(
                edge_index[:,not_self_loop],
                num_nodes, num_parts,
                **partition_kwargs,
            )
        elif algorithm == "pyg-metis":
            node_parts = pyg_metis_partition(
                edge_index[:,not_self_loop],
                num_nodes, num_parts,
            )
        else:
            raise ValueError(f"unknown partition algorithm: {algorithm}")
        
        root_path = Path(root).expanduser().resolve()
        base_path = root_path / f"{algorithm}_{num_parts}"

        if base_path.exists():
            logging.warning(f"directory '{base_path.__str__()}' exists, and will be removed.")
            shutil.rmtree(base_path.__str__())
        base_path.mkdir(parents=True)

        if include_node_attrs is None:
            include_node_attrs = self.node().keys()
        
        if include_edge_attrs is None:
            include_edge_attrs = self.edge().keys()
        
        if include_meta_attrs is None:
            include_meta_attrs = self.meta().keys()

        if ignore_node_attrs is None:
            ignore_node_attrs = set()
        else:
            ignore_node_attrs = set(ignore_node_attrs)
        
        if ignore_edge_attrs is None:
            ignore_edge_attrs = set()
        else:
            ignore_edge_attrs = set(ignore_edge_attrs)
        
        if ignore_meta_attrs is None:
            ignore_meta_attrs = set()
        else:
            ignore_meta_attrs = set(ignore_meta_attrs)
        
        for i in range(num_parts):
            npart_mask = node_parts == i
            epart_mask = npart_mask[edge_index[1]]

            raw_dst_ids: Tensor = torch.where(npart_mask)[0]
            local_edges = edge_index[:, epart_mask]

            raw_src_ids, local_edges = init_vc_edge_index(
                raw_dst_ids, local_edges, bipartite=True,
            )

            g = GraphData.from_bipartite(
                local_edges,
                raw_src_ids=raw_src_ids,
                raw_dst_ids=raw_dst_ids,
            )

            for key in include_node_attrs:
                if key in ignore_node_attrs:
                    continue
                g.node("dst")[key] = self.node()[key][npart_mask]
            
            for key in include_edge_attrs:
                if key in ignore_edge_attrs:
                    continue
                g.edge()[key] = self.edge()[key][epart_mask]
            
            for key in include_meta_attrs:
                if key in ignore_meta_attrs:
                    continue
                g.meta()[key] = self.meta()[key]

            logging.info(f"saving partition data: {i+1}/{num_parts}")
            torch.save(g, (base_path / f"{i:03d}").__str__())
            

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
