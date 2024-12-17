import random
import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data
from starrygl.sample.graph_core import DataSet, DistributedGraphStore

def get_link_prediction_data(data_name: str,  val_ratio, test_ratio):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    #graph_df = pd.read_csv('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edges.csv')
    #if os.path.exists('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/node_features.pt'):
    #    n_feat = torch.load('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/node_features.pt')
    #else:
    #    n_feat = None
    #if os.path.exists('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edge_features.pt'):
    #    e_feat = torch.load('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edge_features.pt')
    #else:
    #    e_feat = None
#
    ## get the timestamp of validate and test set
    #src_node_ids = torch.from_numpy(np.array(graph_df.src.values)).long()
    #dst_node_ids = torch.from_numpy(np.array(graph_df.dst.values)).long()
    #node_interact_times = torch.from_numpy(np.array(graph_df.time.values)).long()
#
    #train_mask = (torch.from_numpy(np.array(graph_df.ext_roll.values)) == 0)
    #test_mask = (torch.from_numpy(np.array(graph_df.ext_roll.values)) == 1)
    #val_mask = (torch.from_numpy(np.array(graph_df.ext_roll.values)) == 2)
    # the setting of seed follows previous works
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(data_name, data_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(data_name, data_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(data_name, data_name))
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {data_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {data_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
        e_feat = edge_raw_features
    n_feat = torch.from_numpy(node_raw_features.astype(np.float32))
    e_feat = torch.from_numpy(edge_raw_features.astype(np.float32))
    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    src_node_ids = torch.from_numpy(graph_df.u.values.astype(np.longlong))
    dst_node_ids = torch.from_numpy(graph_df.i.values.astype(np.longlong))
    node_interact_times = torch.from_numpy(graph_df.ts.values.astype(np.float32))
    #edge_ids = torch.from_numpy(graph_df.idx.values.astype(np.longlong))
    labels = torch.from_numpy(graph_df.label.values)
    unique_node_ids = torch.cat((src_node_ids,dst_node_ids)).unique()
    train_mask = node_interact_times <= val_time
    val_mask = ((node_interact_times > val_time)&(node_interact_times <= test_time))
    test_mask = (node_interact_times > test_time)
    torch.manual_seed(2020)
    train_node_set = torch.cat((src_node_ids[train_mask],dst_node_ids[train_mask])).unique()
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * unique_node_ids.shape[0])))

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = torch.from_numpy(np.logical_and(~new_test_source_mask, ~new_test_destination_mask)).long()
    train_mask = (train_mask & observed_edges_mask)
    mask = torch.isin(unique_node_ids,train_node_set,invert = True)
    new_node_set = unique_node_ids[mask]
    edge_contains_new_node_mask = (torch.isin(src_node_ids,new_node_set) | torch.isin(dst_node_ids,new_node_set))
    new_node_val_mask = (val_mask & edge_contains_new_node_mask)
    new_node_test_mask = (test_mask & edge_contains_new_node_mask)
    full_data = Data()
    full_data.edge_index = torch.stack((src_node_ids,dst_node_ids))
    sample_graph = {}
    sample_src = torch.cat([src_node_ids.view(-1, 1), dst_node_ids.view(-1, 1)], dim=1)\
        .reshape(1, -1)
    sample_dst = torch.cat([dst_node_ids.view(-1, 1), src_node_ids.view(-1, 1)], dim=1)\
        .reshape(1, -1)
    sample_ts = torch.cat([node_interact_times.view(-1, 1), node_interact_times.view(-1, 1)], dim=1).reshape(-1)
    sample_eid = torch.arange(full_data.edge_index.shape[1]).view(-1, 1).repeat(1, 2).reshape(-1)
    sample_graph['edge_index'] = torch.cat([sample_src, sample_dst], dim=0)
    sample_graph['ts'] = sample_ts
    sample_graph['eids'] = sample_eid
    sample_graph['train_mask'] = train_mask
    sample_graph['val_mask'] = val_mask
    sample_graph['test_mask'] = val_mask
    sample_graph['new_node_val_mask'] = new_node_val_mask
    sample_graph['new_node_test_mask'] = new_node_test_mask
    print(unique_node_ids.max().item(),unique_node_ids.shape[0])
    full_data.num_nodes = int(unique_node_ids.max().item())+1
    full_data.num_edges = node_interact_times.shape[0]
    full_data.sample_graph = sample_graph
    full_data.x = n_feat
    full_data.edge_attr = e_feat
    full_data.y = labels
    full_data.edge_ts = node_interact_times
    full_data.train_mask = train_mask
    full_data.val_mask = val_mask
    full_data.test_mask = test_mask
    full_data.new_node_val_mask = new_node_val_mask
    full_data.new_node_test_mask = new_node_test_mask
    return full_data
    #full_graph = DistributedGraphStore(full_data, device, uvm_node, uvm_edge)
    #train_data = torch.masked_select(full_data.edge_index,train_mask.to(device)).reshape(2,-1)
    #train_ts = torch.masked_select(full_data.edge_ts,train_mask.to(device))
    #val_data = torch.masked_select(full_data.edge_index,val_mask.to(device)).reshape(2,-1)
    #val_ts = torch.masked_select(full_data.edge_ts,val_mask.to(device))
    #test_data = torch.masked_select(full_data.edge_index,test_mask.to(device)).reshape(2,-1)
    #test_ts = torch.masked_select(full_data.edge_ts,test_mask.to(device)) 
    ##print(train_data.shape[1],val_data.shape[1],test_data.shape[1])
    #train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(train_mask).view(-1))
    #test_data = DataSet(edges = test_data,ts =test_ts,eids = torch.nonzero(test_mask).view(-1))
    #val_data = DataSet(edges = val_data,ts = val_ts,eids = torch.nonzero(val_mask).view(-1))
    #new_node_val_data = torch.masked_select(full_data.edge_index,new_node_val_mask.to(device)).reshape(2,-1)
    #new_node_val_ts = torch.masked_select(full_data.edge_ts,new_node_val_mask.to(device))
    #new_node_test_data = torch.masked_select(full_data.edge_index,new_node_test_mask.to(device)).reshape(2,-1)
    #new_node_test_ts = torch.masked_select(full_data.edge_ts,new_node_test_mask.to(device)) 
    #return  full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data
