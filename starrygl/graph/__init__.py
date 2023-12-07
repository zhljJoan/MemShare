from .route import Route
from .utils import init_vc_edge_index

from torch import Tensor
from typing import Tuple

__all__ = [
    "Route",
    "init_vc_edge_index",
    "new_vc_route",
]

def new_vc_route(
    dst_ids: Tensor,
    edge_index: Tensor,
    bipartite: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Route]:
    src_ids, local_edge_index = init_vc_edge_index(
        dst_ids, edge_index, bipartite=bipartite)
    
    route = Route.from_raw_indices(
        src_ids, dst_ids, bipartite=bipartite)
    
    return src_ids, local_edge_index, dst_ids, route
