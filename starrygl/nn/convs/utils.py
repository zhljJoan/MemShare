import torch

from torch import Tensor
from typing import *

from starrygl.graph import DistGraph


class ShrinkData:
    def __init__(self, g: DistGraph, m: Tensor) -> None:
        edge_index = g.edge_index
        
        # src_to_dst edge's mask
        m = m[edge_index[0]]
        
        # dst's mask
        m = edge_index[1][m]
        dst_m: Tensor = torch.zeros(g.dst_size, dtype=torch.bool, device=m.device).index_fill_(0, m, 1)
        self.dst_mask = dst_m
        
        # dst_to_src edge's mask
        m = dst_m[edge_index[1]]
        self.edge_mask = m
        
        # src's mask
        edge_index = edge_index[:,m]
        src_m: Tensor = torch.zeros(g.src_size, dtype=torch.bool, device=m.device).index_fill_(0, edge_index[0], 1)
        self.src_mask = src_m
        
        self._src_size = src_m.count_nonzero()
        self._dst_size = dst_m.count_nonzero()
        
        # print(f"{dst_m.size()}_{self._dst_size}")
        
        # edge_index
        imp = torch.empty(g.src_size, dtype=torch.long, device=m.device)
        imp[src_m] = torch.arange(self._src_size, dtype=torch.long, device=m.device)
        src = imp[edge_index[0]]
        
        imp = torch.empty(g.dst_size, dtype=torch.long, device=m.device)
        imp[dst_m] = torch.arange(self._dst_size, dtype=torch.long, device=m.device)
        dst = imp[edge_index[1]]
        
        self.edge_index = torch.vstack([src, dst])
    
    @property
    def src_size(self) -> int:
        return self._src_size.item()
    
    @property
    def dst_size(self) -> int:
        return self._dst_size.item()
        