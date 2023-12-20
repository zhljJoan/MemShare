# import torch

# from torch import Tensor
# from typing import *

# from torch_scatter import scatter_sum
# from starrygl.core.route import Route


# def compute_in_degree(edge_index: Tensor, route: Route) -> Tensor:
#     dst_size = route.src_size
#     x = torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    
#     in_deg = scatter_sum(x, edge_index[1], dim=0, dim_size=dst_size)
#     in_deg, _ = route.forward_a2a(in_deg)
#     return in_deg

# def compute_out_degree(edge_index: Tensor, route: Route) -> Tensor:
#     src_size = route.dst_size
#     x = torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device)

#     out_deg = scatter_sum(x, edge_index[0], dim=0, dim_size=src_size)
#     out_deg, _ = route.backward_a2a(out_deg)
#     out_deg, _ = route.forward_a2a(out_deg)

#     return out_deg

# def compute_gcn_norm(edge_index: Tensor, route: Route) -> Tensor:
#     in_deg = compute_in_degree(edge_index, route)
#     out_deg = compute_out_degree(edge_index, route)
    
#     a = in_deg[edge_index[0]].pow(-0.5)
#     b = out_deg[edge_index[0]].pow(-0.5)
#     x = a * b
#     x[x.isinf()] = 0.0
#     x[x.isnan()] = 0.0
#     return x
