import pandas as pd
import numpy as np
import os
import torch
def get_link_prediction_data(data_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edges.csv')
    if os.path.exists('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/node_features.pt'):
        n_feat = torch.load('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/node_features.pt')
    else:
        n_feat = None
    if os.path.exists('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edge_features.pt'):
        e_feat = torch.load('/mnt/nfs/fzz/TGL-DATA/'+data_name+'/edge_features.pt')
    else:
        e_feat = None

    # get the timestamp of validate and test set
    src_node_ids = torch.from_numpy(np.array(graph_df.src.values)).long()
    dst_node_ids = torch.from_numpy(np.array(graph_df.dst.values)).long()
    node_interact_times = torch.from_numpy(np.array(graph_df.time.values)).long()

    train_mask = (torch.from_numpy(np.array(graph_df.ext_roll.values)) == 0)
    test_mask = (torch.from_numpy(np.array(graph_df.ext_roll.values)) == 1)
    val_mask = (torch.from_numpy(np.array(graph_df.ext_roll.values)) == 2)
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    # the setting of seed follows previous works
    unique_node_ids = torch.cat((src_node_ids,dst_node_ids)).unique()
    torch.manual_seed(2020)
    test_node_set = torch.cat((src_node_ids[test_mask],dst_node_ids[test_mask])).unique()
    train_node_set = torch.cat((src_node_ids[train_mask],dst_node_ids[train_mask])).unique()
    new_test_node_set = torch.randint(0,test_node_set.shape[0],int(0.1*unique_node_ids.shape[0]))
    new_test_node_set = test_node_set[new_test_node_set]
    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.src.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.dst.map(lambda x: x in new_test_node_set).values
    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = torch.from_numpy(np.logical_and(~new_test_source_mask, ~new_test_destination_mask)).long()
    mask = torch.isin(unique_node_ids,train_node_set,invert = True)
    new_node_set = unique_node_ids[mask]
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])


    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data
