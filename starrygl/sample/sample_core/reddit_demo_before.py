import argparse
import random
import pandas as pd
import numpy as np
import torch
import time

from tqdm import tqdm
from .Utils import GraphData

class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

class NegLinkInductiveSampler:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, n):
        return np.random.choice(self.nodes, size=n)

def load_reddit_dataset():
    df = pd.read_csv('/mnt/data/hzq/DATA/{}/edges.csv'.format("REDDIT"))
    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    src = torch.tensor(df['src'].to_numpy(dtype=int))
    dst = torch.tensor(df['dst'].to_numpy(dtype=int))
    edge_index = torch.stack([src, dst])
    timestamp = torch.tensor(df['time']).float()
    g = GraphData(0, edge_index, timestamp=timestamp, data=None, partptr=torch.tensor([0, num_nodes]))
    return g, df

def load_gdelt_dataset():
    df = pd.read_csv('/mnt/data/hzq/DATA/{}/edges.csv'.format("GDELT"))
    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    src = torch.tensor(df['src'].to_numpy(dtype=int))
    dst = torch.tensor(df['dst'].to_numpy(dtype=int))
    edge_index = torch.stack([src, dst])
    timestamp = torch.tensor(df['time']).float()
    g = GraphData(0, edge_index, timestamp=timestamp, data=None, partptr=torch.tensor([0, num_nodes]))
    return g, df


def test():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name',default="REDDIT")
    parser.add_argument('--config', type=str, help='path to config file',default="/home/zlj/hzq/project/code/TGL/config/TGN.yml")
    parser.add_argument('--batch_size', type=int, default=600, help='path to config file')
    parser.add_argument('--num_thread', type=int, default=64, help='number of thread')
    args=parser.parse_args()
    dataset = "gdelt"#"reddit"#"gdelt"

    seed=10
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g_data, df = load_reddit_dataset()
    print(g_data)
    # for worker in [1,2,3,4,5,6,7,8,9,10,20,30]:
    # import random
    # timestamp = [random.randint(1, 5) for i in range(0, g.num_edges)]
    # timestamp = torch.FloatTensor(timestamp)

    # print('begin load')
    # pre = time.time()
    # # timestamp = torch.load('/home/zlj/hzq/code/gnn/my_sampler/TemporalSample/timestamp.my')
    # tnb = torch.load("tnb_reddit_before.my")
    # end = time.time()
    # print("load time:", end-pre)
    # row, col = g.edge_index
    edge_weight=None
    # g_data = GraphData(id=1, edge_index=g.edge_index, timestamp=timestamp, data=g, partptr=torch.tensor([0, g.num_nodes//4, g.num_nodes//4*2, g.num_nodes//4*3, g.num_nodes]))

    from .neighbor_sampler import NeighborSampler, SampleType, get_neighbors

    print('begin tnb')
    row, col = g_data.edge_index
    row = torch.cat([row, col])
    col = torch.cat([col, row])
    eid = torch.cat([g_data.eid, g_data.eid])
    timestamp = torch.cat([g_data.edge_ts, g_data.edge_ts])
    timestamp,ind = timestamp.sort()
    timestamp = timestamp.float().contiguous()
    eid  = eid[ind].contiguous()
    row = row[ind]
    col = col[ind]
    print(row, col)
    g2 = GraphData(0, torch.stack([row, col]), timestamp=timestamp, data=None, partptr=torch.tensor([0, max(int(df['src'].max()), int(df['dst'].max())) + 1]))
    print(g2)
    pre = time.time()
    tnb = get_neighbors(dataset, row.contiguous(), col.contiguous(), g_data.num_nodes, 0, eid, edge_weight, timestamp)
    end = time.time()
    print("init tnb time:", end-pre)
    # torch.save(tnb, "tnb_{}_before.my".format(dataset), pickle_protocol=4)


    pre = time.time()
    sampler = NeighborSampler(g_data.num_nodes, 
                            tnb=tnb,
                            num_layers=1, 
                            fanout=[10], 
                            graph_data=g_data, 
                            workers=32, 
                            policy="recent", 
                            graph_name='a')
    end = time.time()
    print("init time:", end-pre)

    # neg_link_sampler = NegLinkSampler(g_data.num_nodes)
    from .base import NegativeSampling, NegativeSamplingMode
    neg_link_sampler = NegativeSampling(NegativeSamplingMode.triplet)

    # from torch_geometric.sampler import NeighborSampler, NumNeighbors, NodeSamplerInput, SamplerOutput
    # pre = time.time()
    # num_nei = NumNeighbors([100, 100])
    # node_idx = NodeSamplerInput(input_id=None, node=torch.tensor(range(g.num_nodes//4, g.num_nodes//4+600000)))# (input_id=None, node=torch.masked_select(torch.arange(g.num_nodes),node_data['train_mask']))
    # sampler = NeighborSampler(g, num_nei)
    # end = time.time()
    # print("init time:", end-pre)


    out = []
    tot_time = 0
    sam_time = 0
    sam_edge = 0
    pre = time.time()

    min_than_ten = 0
    min_than_ten_sum = 0
    seed_node_sum = 0
    for _, rows in tqdm(df.groupby(df.index // args.batch_size), total=len(df) // args.batch_size):
            # root_nodes = torch.tensor(np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))])).long()
            # ts = torch.tensor(np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32))
            # outi = sampler.sample_from_nodes(root_nodes, ts=ts)
            edges = torch.tensor(np.stack([rows.src.values, rows.dst.values])).long()
            outi, meta = sampler.sample_from_edges(edges=edges, ets=torch.tensor(rows.time.values).float(), neg_sampling=neg_link_sampler)
            # min_than_ten += (torch.tensor(tnb.deg)[meta['seed']]<10).sum()
            # min_than_ten_sum += ((torch.tensor(tnb.deg)[meta['seed']])[torch.tensor(tnb.deg)[meta['seed']]<10]).sum()
            # seed_node_sum += meta['seed'].size(0)
            tot_time += outi[0].tot_time
            sam_time += outi[0].sample_time
            # print(outi[0].sample_edge_num)
            sam_edge += outi[0].sample_edge_num
            # out.append(outi)

    end = time.time()
    # print("row", out[23][0].row())
    print("sample time", end-pre)
    print("tot_time", tot_time)
    print("sam_time", sam_time)
    print("sam_edge", sam_edge)
    # print('eid_list:', out[23][0].eid())
    # print('delta_ts_list:', out[10][0].delta_ts)
    # print('node:', out[23][0].sample_nodes())
    # print('node_ts:', out[23][0].sample_nodes_ts)
    # print('eid_list:', out[23][1].eid)
    # print('node:', out[23][1].sample_nodes)
    # print('node_ts:', out[23][1].sample_nodes_ts)
    # print('edge_index_list:', out[0][0].edge_index)
    # print("min_than_ten", min_than_ten)
    # print("min_than_ten_sum", min_than_ten_sum)
    # print("seed_node_sum", seed_node_sum)
    # print("predict edge_num", (seed_node_sum-min_than_ten)*9+min_than_ten_sum)
    print('吞吐量  : {:.4f}'.format(sam_edge/(end-pre)))
    
if __name__ == "__main__":
    test()