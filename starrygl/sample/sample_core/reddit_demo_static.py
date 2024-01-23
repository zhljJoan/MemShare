import argparse
import random
import pandas as pd
import numpy as np
import torch
import time

from tqdm import tqdm
from .Utils import GraphData

def load_reddit_dataset():
    df = pd.read_csv('/mnt/data/hzq/DATA/{}/edges.csv'.format("REDDIT"))
    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    src = torch.tensor(df['src'].to_numpy(dtype=int))
    dst = torch.tensor(df['dst'].to_numpy(dtype=int))
    row = torch.cat([src, dst])
    col = torch.cat([dst, src])
    edge_index = torch.stack([row, col])
    timestamp = torch.tensor(df['time']).float()
    g = GraphData(0, edge_index, timestamp=None, data=None, partptr=torch.tensor([0, num_nodes]))
    return g

def load_gdelt_dataset():
    df = pd.read_csv('/mnt/data/hzq/DATA/{}/edges.csv'.format("GDELT"))
    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    src = torch.tensor(df['src'].to_numpy(dtype=int))
    dst = torch.tensor(df['dst'].to_numpy(dtype=int))
    row = torch.cat([src, dst])
    col = torch.cat([dst, src])
    edge_index = torch.stack([row, col])
    timestamp = torch.tensor(df['time']).float()
    g = GraphData(0, edge_index, timestamp=None, data=None, partptr=torch.tensor([0, num_nodes]))
    return g

def load_ogb_dataset():
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name='ogbn-products', root="/home/zlj/hzq/code/gnn/dataset/")
    split_idx = dataset.get_idx_split()
    g = dataset[0]
    n_node = g.num_nodes
    node_data={}
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    src, dst = g.edge_index
    row = torch.cat([src, dst])
    col = torch.cat([dst, src])
    edge_index = torch.stack([row, col])
    g = GraphData(id=0, edge_index=edge_index, data=g, partptr=torch.tensor([0, g.num_nodes]))
    return g # , node_data


def test():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name',default="REDDIT")
    # parser.add_argument('--config', type=str, help='path to config file',default="/home/zlj/hzq/project/code/TGL/config/TGN.yml")
    parser.add_argument('--batch_size', type=int, default=600, help='path to config file')
    # parser.add_argument('--num_thread', type=int, default=64, help='number of thread')
    args=parser.parse_args()

    seed=10
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g_data = load_reddit_dataset()
    print(g_data)

    from .neighbor_sampler import NeighborSampler, get_neighbors

    # print('begin tnb')
    # row, col = g_data.edge_index
    # pre = time.time()
    # tnb = get_neighbors("a", 
    #                     row.contiguous(), 
    #                     col.contiguous(), 
    #                     g_data.num_nodes, 0, 
    #                     g_data.eid, 
    #                     None, 
    #                     None)
    # end = time.time()
    # print("init tnb time:", end-pre)

    pre = time.time()
    sampler = NeighborSampler(g_data.num_nodes,
                            num_layers=2, 
                            fanout=[10,10], 
                            graph_data=g_data, 
                            workers=32, 
                            policy="uniform", 
                            graph_name='a',
                            is_distinct=0)
    end = time.time()
    print("init time:", end-pre)
    print(sampler.tnb.deg[0:100])
    
    n_list = []
    for i in range (g_data.num_nodes.item() // args.batch_size):
        if i+args.batch_size< g_data.num_nodes.item():
            n_list.append(range(i*args.batch_size, i*args.batch_size+args.batch_size))
        else:
            n_list.append(range(i*args.batch_size, g_data.num_nodes.item()))
    # print(n_list)

    out = []
    tot_time = 0
    sam_time = 0
    sam_edge = 0
    sam_node = 0
    pre = time.time()
    for i, nodes in tqdm(enumerate(n_list), total=g_data.num_nodes.item() // args.batch_size):
    # for nodes in n_list:
        root_nodes = torch.tensor(nodes).long()
        outi = sampler.sample_from_nodes(root_nodes)
        sam_node += outi[0].sample_nodes().size(0)
        sam_node += outi[1].sample_nodes().size(0)
        sam_edge += outi[0].sample_edge_num

    end = time.time()
    print("sample time", end-pre)
    print("sam_edge", sam_edge)
    print("sam_node", sam_node)
    print('边吞吐量  : {:.4f}'.format(sam_edge/(end-pre)))
    print('点吞吐量  : {:.4f}'.format(sam_node/(end-pre)))
    
if __name__ == "__main__":
    test()