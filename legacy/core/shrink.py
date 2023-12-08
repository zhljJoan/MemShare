# import torch

# from torch import Tensor
# from typing import *



# class ShrinkData:
#     no_idx: int = (2**62-1)*2+1
    
#     def __init__(self,
#         src_size: int,
#         dst_size: int,
#         dst_idx: Tensor,
#         edge_index: Tensor,
#         bipartite: bool = False,
#     ) -> None:
#         device = dst_idx.device
#         tmp = torch.empty(max(src_size, dst_size), dtype=torch.bool, device=device)
        
#         tmp.fill_(0)
#         tmp.index_fill_(0, dst_idx, 1)
        
#         edge_idx = torch.where(tmp[edge_index[1]])[0]
#         edge_index = edge_index[:, edge_idx]
        
#         if bipartite:
#             tmp.fill_(0)
#             tmp.index_fill_(0, edge_index[0], 1)
#             src_idx = torch.where(tmp)[0]

#             imp = torch.empty(max(src_size, dst_size), dtype=torch.long, device=device)
#             imp[dst_idx] = torch.arange(dst_idx.size(0), dtype=torch.long, device=device)
#             dst = imp[edge_index[1]]
            
#             imp.fill_(self.no_idx)
#             imp[src_idx] = torch.arange(src_idx.size(0), dtype=torch.long, device=device)
#             src = imp[edge_index[0]]
#             edge_index = torch.vstack([src, dst])
#         else:
#             tmp.index_fill_(0, edge_index[0], 1)
#             tmp.index_fill_(0, dst_idx, 0)
#             src_idx = torch.cat([dst_idx, torch.where(tmp)[0]], dim=0)

#             imp = torch.empty(max(src_size, dst_size), dtype=torch.long, device=device)
#             imp.fill_(self.no_idx)
#             imp[src_idx] = torch.arange(src_idx.size(0), dtype=torch.long, device=device)
#             edge_index = imp[edge_index.flatten()].view_as(edge_index)
        
#         self.src_idx = src_idx
#         self.dst_idx = dst_idx
#         self.edge_idx = edge_idx
#         self.edge_index = edge_index
#         self._src_imp = imp[:src_size]
    
#     def to(self, device):
#         self.src_idx = self.src_idx.to(device)
#         self.dst_idx = self.dst_idx.to(device)
#         self.edge_idx = self.edge_idx.to(device)
#         self.edge_index = self.edge_index.to(device)
#         self._src_imp = self._src_imp.to(device)
#         return self
    
#     def shrink_src_val_and_idx(self, val: Tensor, idx: Tensor) -> Tuple[Tensor, Tensor]:
#         idx = self._src_imp[idx]
#         m = (idx != self.no_idx)
#         return val[m], idx[m]
        
#     @property
#     def src_size(self) -> int:
#         return self.src_idx.size(0)
    
#     @property
#     def dst_size(self) -> int:
#         return self.dst_idx.size(0)
    
#     @property
#     def edge_size(self) -> int:
#         return self.edge_idx.size(0)
    
