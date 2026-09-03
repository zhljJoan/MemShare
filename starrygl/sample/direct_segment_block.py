from dgl import heterograph_index
from dgl.heterograph import DGLBlock


def create_block(row, col, num_src, num_dst):
    """Build direct COO once and retain DGL's CSC segment plan."""
    if row.device != col.device or row.dtype != col.dtype or row.shape != col.shape:
        raise ValueError("row and col must be matching vectors")
    graph_index = heterograph_index.create_unitgraph_from_coo(
        2, int(num_src), int(num_dst), row, col, ["coo", "csc"], False, False
    )
    block = DGLBlock(graph_index, (["_N"], ["_N"]), ["_E"])
    dst_ptr, _, edge_ids = block.adj_tensors("csc")
    block._memshare_edge_src = row
    block._memshare_segment_plan = (dst_ptr, edge_ids)
    return block
