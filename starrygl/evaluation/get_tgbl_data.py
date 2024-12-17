from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.data import Data
import torch

def get_tgbl_prediction_data(name):
# data loading
    dataset = PyGLinkPropPredDataset(name=name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()
    metric = dataset.eval_metric
    src_node_ids = data.src
    dst_node_ids = data.dst
    node_interact_times = data.t
    e_feat = data.msg
    labels = data.y
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
    sample_graph['test_mask'] = test_mask
    unique_node_ids = torch.unique(torch.cat((src_node_ids,dst_node_ids)))
    print(unique_node_ids.max().item(),unique_node_ids.shape[0])
    full_data.num_nodes = int(unique_node_ids.max().item())+1
    full_data.num_edges = node_interact_times.shape[0]
    full_data.sample_graph = sample_graph
    full_data.edge_attr = e_feat
    full_data.y = labels
    full_data.edge_ts = node_interact_times
    full_data.train_mask = train_mask
    full_data.val_mask = val_mask
    full_data.test_mask = test_mask
    return full_data
    
if __name__ == "__main__":
    name = 'tgbl-wiki'
    get_tgbl_prediction_data(name)