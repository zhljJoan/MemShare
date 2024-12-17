import torch
import time
from Utils import GraphData
seed = 10  # 你可以选择任何整数作为种子
torch.manual_seed(seed)

num_nodes1 = 10
fanout1 = [2]
edge_index1 = torch.tensor([[1, 5, 7, 9, 2, 4, 6, 7, 8, 0, 1, 6, 2, 0, 1, 3, 5, 8, 9, 7, 4, 8, 2, 3, 5, 8],
                            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 6, 6, 7, 7, 8, 9]])
edge_ts =      torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6]).double()
edge_weight1 = torch.tensor([2, 1, 2, 1, 8, 6, 3, 1, 1, 1, 1, 5, 1, 1, 2, 1, 1, 1, 1, 5, 1, 2, 2, 2, 1, 1]).double()
edge_weight1 = None

g_data = GraphData(id=0, edge_index=edge_index1, timestamp=edge_ts, data=None, partptr=torch.tensor([0, num_nodes1]))
from neighbor_sampler import NeighborSampler, SampleType
pre = time.time()
# from neighbor_sampler import get_neighbors, update_edge_weight
# row, col = edge_index1
# tnb = get_neighbors(row.contiguous(), col.contiguous(), num_nodes1, edge_weight1)
# print("tnb.neighbors:", tnb.neighbors)
# print("tnb.deg:", tnb.deg)
# print("tnb.weight:", tnb.edge_weight)
sampler = NeighborSampler(num_nodes1, 
                          num_layers=1, 
                          fanout=fanout1, 
                          edge_weight=edge_weight1, 
                          graph_data=g_data, 
                          workers=2, 
                          graph_name='a',
                          is_distinct = 0,
                          policy="recent")
end = time.time()
print("init time:", end-pre)
print("tnb.neighbors:", sampler.tnb.neighbors)
print("tnb.deg:", sampler.tnb.deg)
print("tnb.ts:", sampler.tnb.timestamp)
print("tnb.weight:", sampler.tnb.edge_weight)


# update_edge_row =     row
# update_edge_col =     col
# update_edge_w =  torch.DoubleTensor([i for i in range(edge_weight1.size(0))])
# print('tnb.edge_weight:', tnb.edge_weight)
# print('begin update')
# pre = time.time()
# update_edge_weight(tnb, update_edge_row.contiguous(), update_edge_col.contiguous(), update_edge_w.contiguous())
# end = time.time()
# print("update time:", end-pre)
# print('update_edge_row:', update_edge_row)
# print('update_edge_col:', update_edge_col)
# print('tnb.edge_weight:', tnb.edge_weight)


pre = time.time()

out = sampler.sample_from_nodes(torch.tensor([6,7]),
                                 with_outer_sample=SampleType.Whole, 
                                 ts=torch.tensor([9, 9]))
end = time.time()
# print('node:', out.node)
# print('edge_index_list:', out.edge_index_list)
# print('eid_list:', out.eid_list)
# print('eid_ts_list:', out.eid_ts_list)
print("sample time:", end-pre)
print("tot_time", out[0].tot_time)
print("sam_time", out[0].sample_time)
print("sam_edge", out[0].sample_edge_num)
print('eid_list:', out[0].eid)
print('delta_ts_list:', out[0].delta_ts)
print((out[0].sample_nodes<10000).sum())
print('node:', out[0].sample_nodes)
print('node_ts:', out[0].sample_nodes_ts)