from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_remaining_self_loops, to_undirected

import os.path as osp
import sys

from starrygl.data import GraphData

import logging
logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    data = Planetoid("/mnt/nfs/hwj/pyg_datasets/Planetoid/Cora", "Cora")[0]
    if data.is_directed():
        data.edge_index, _ = to_undirected(data.edge_index)
    data.edge_index, _ = add_remaining_self_loops(data.edge_index)
    
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges: {data.num_edges}")
    print(f"num_features: {data.num_features}")

    data = GraphData.from_pyg_data(data)

    num_parts_list = [1, 2, 3, 5, 7, 9, 11]
    algos = ["metis", 'mt-metis', "random"]
    
    root = osp.splitext(osp.abspath(__file__))[0]
    print(f"root: {root}")
    for num_parts in num_parts_list:
        for algo in algos:
            print(f"======== {num_parts} + {algo} ========")
            data.save_partition(root, num_parts, algo)
