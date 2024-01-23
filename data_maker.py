
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from starrygl.sample.sample_core.base import NegativeSampling
from starrygl.sample.part_utils.partition_tgnn import partition_save
parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--data_name', default='WIKI', type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--num_neg_sample', default=1, type=int, metavar='W',
                    help='number of negative samples')
args = parser.parse_args()


def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            edge_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats


data_name = args.data_name
g = np.load('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/ext_full.npz')
df = pd.read_csv('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edges.csv')
if os.path.exists('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/node_features.pt'):
    n_feat = torch.load('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/node_features.pt')
else:
    n_feat = None
if os.path.exists('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edge_features.pt'):
    e_feat = torch.load('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edge_features.pt')
else:
    e_feat = None

src = torch.from_numpy(np.array(df.src.values)).long()
dst = torch.from_numpy(np.array(df.dst.values)).long()
ts = torch.from_numpy(np.array(df.time.values)).long()
neg_nums = args.num_neg_sample

edge_index = torch.cat((src[np.newaxis, :], dst[np.newaxis, :]), 0)
num_nodes = edge_index.view(-1).max().item()+1
num_edges = edge_index.shape[1]
print('the number of nodes in graph is {}, \
      the number of edges in graph is {}'.format(num_nodes, num_edges))
sample_graph = {}
sample_src = torch.cat([src.view(-1, 1), dst.view(-1, 1)], dim=1)\
    .reshape(1, -1)
sample_dst = torch.cat([dst.view(-1, 1), src.view(-1, 1)], dim=1)\
    .reshape(1, -1)
sample_ts = torch.cat([ts.view(-1, 1), ts.view(-1, 1)], dim=1).reshape(-1)
sample_eid = torch.arange(num_edges).view(-1, 1).repeat(1, 2).reshape(-1)
sample_graph['edge_index'] = torch.cat([sample_src, sample_dst], dim=0)
sample_graph['ts'] = sample_ts
sample_graph['eids'] = sample_eid
neg_sampler = NegativeSampling('triplet')
neg_src = neg_sampler.sample(edge_index.shape[1]*neg_nums, num_nodes)
neg_sample = neg_src.reshape(-1, neg_nums)


edge_ts = torch.torch.from_numpy(np.array(ts)).float()
data = Data()
data.num_nodes = num_nodes
data.num_edges = num_edges

data.edge_index = edge_index
data.edge_ts = edge_ts
data.neg_sample = neg_sample

if n_feat is not None:
    data.x = n_feat
if e_feat is not None:
    data.edge_attr = e_feat

data.train_mask = (torch.from_numpy(np.array(df.ext_roll.values)) == 0)
data.val_mask = (torch.from_numpy(np.array(df.ext_roll.values)) == 1)
data.test_mask = (torch.from_numpy(np.array(df.ext_roll.values)) == 2)
print(ts[data.train_mask].min(),ts[data.train_mask].max(),ts[data.val_mask].min(),ts[data.val_mask].max(),ts[data.test_mask].min(),ts[data.test_mask].max())
sample_graph['train_mask'] = data.train_mask[sample_eid]
sample_graph['test_mask'] = data.test_mask[sample_eid]
sample_graph['val_mask'] = data.val_mask[sample_eid]
data.sample_graph = sample_graph

data.y = torch.zeros(edge_index.shape[1])
edge_index_dict = {}
edge_index_dict['edata'] = data.edge_index
edge_index_dict['sample_data'] = data.sample_graph['edge_index']
#edge_index_dict['neg_data'] = torch.cat([neg_src.view(1, -1),


#                                         dst.view(-1, 1).repeat(1, neg_nums).
#                                         reshape(1, -1)], dim=0)
data.edge_index_dict = edge_index_dict
edge_weight_dict = {}
edge_weight_dict['edata'] = 1*neg_nums
edge_weight_dict['sample_data'] = 1*neg_nums
#edge_weight_dict['neg_data'] = 1
partition_save('/mnt/data/part_data/v2/here/'+data_name, data, 1, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
partition_save('/mnt/data/part_data/v2/here/'+data_name, data, 2, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
partition_save('/mnt/data/part_data/v2/here/'+data_name, data, 4, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
partition_save('/mnt/data/part_data/v2/here/'+data_name, data, 8, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
#partition_save('./dataset/here/'+data_name, data, 16, 'metis_for_tgnn',
#               edge_weight_dict=edge_weight_dict)
#
# partition_save('./dataset/here/'+data_name, data, 4, 'metis_for_tgnn',
# edge_weight_dict=edge_weight_dict )
# partition_save('./dataset/here'+data_name, data, 8, 'metis')
# partition_save('./dataset/'+data_name, data, 12, 'metis')
# partition_save('./dataset/'+data_name, data, 16, 'metis')

