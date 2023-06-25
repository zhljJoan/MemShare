# import torch

# from torch import Tensor
# from typing import *

# def init_local_edge_index(
#     dst_ids: Tensor,
#     edge_index: Tensor,
#     bipartite: bool = False,
# ) -> Tuple[Tensor, Tensor]:
#     max_ids = calc_max_ids(dst_ids, edge_index)
#     ikw = dict(dtype=torch.long, device=dst_ids.device)
#     xmp = torch.zeros(max_ids + 1, **ikw)

#     # 判断是不是点分割且所有边被划分到目标点所在分区
#     xmp[edge_index[1].unique()] += 0b01
#     xmp[dst_ids.unique()] += 0b10
#     if not (xmp != 0x01).all():
#         raise RuntimeError(f"must be vertex-cut partition graph")
    
#     if bipartite:
#         src_ids = edge_index[0].unique()
#     else:
#         # 假设是同构图
#         # src_ids 等于 [dst_ids, edge_index[0] except dst_ids]
#         xmp.fill_(0)
#         xmp[edge_index[0]] = 1
#         xmp[dst_ids] = 0
#         src_ids = torch.cat([dst_ids, torch.where(xmp > 0)[0]], dim=-1)

#     # 计算局部索引
#     xmp.fill_((2**62-1)*2+1)
#     xmp[src_ids] = torch.arange(src_ids.size(0), **ikw)
#     src = xmp[edge_index[0]]
    
#     xmp.fill_((2**62-1)*2+1)
#     xmp[dst_ids] = torch.arange(dst_ids.size(0), **ikw)
#     dst = xmp[edge_index[1]]
    
#     local_edge_index = torch.vstack([src, dst])
#     return src_ids, local_edge_index

# def calc_max_ids(*ids: Tensor) -> int:
#     x = [t.max().item() if t.numel() > 0 else 0 for t in ids]
#     return max(*x)