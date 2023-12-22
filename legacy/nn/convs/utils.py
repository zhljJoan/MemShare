# import torch

# from torch import Tensor
# from typing import *



# class ShrinkHelper:
#     def __init__(self, g, dst_idx: Tensor) -> None:
#         from starrygl.graph import DistGraph
#         g: DistGraph = g

#         self.device = dst_idx.device
        
#         dst_m = torch.zeros(g.dst_size, dtype=torch.bool, device=self.device)
#         dst_m.index_fill_(0, dst_idx, 1)
        
#         edge_idx = torch.where(dst_m[g.edge_index[1]])[0]
#         edge_index = g.edge_index[:, edge_idx]
        
#         src_idx = edge_index[0]
#         src_m = torch.zeros(g.src_size, dtype=torch.bool, device=self.device)
#         src_m.index_fill_(0, src_idx, 1)
#         src_idx = torch.where(src_m)[0]
        
#         imp = torch.empty(max(g.src_size, g.dst_size), dtype=torch.long, device=self.device)
        
#         imp[src_idx] = torch.arange(src_idx.size(0), dtype=torch.long, device=self.device)
#         src = imp[edge_index[0]]
        
#         imp[dst_idx] = torch.arange(dst_idx.size(0), dtype=torch.long, device=self.device)
#         dst = imp[edge_index[1]]
        
#         self.src_idx = src_idx
#         self.dst_idx = dst_idx
#         self.edge_idx = edge_idx
#         self.edges = torch.vstack([src, dst])
        
#     @property
#     def src_size(self) -> int:
#         return self.src_idx.size(0)
    
#     @property
#     def dst_size(self) -> int:
#         return self.dst_idx.size(0)