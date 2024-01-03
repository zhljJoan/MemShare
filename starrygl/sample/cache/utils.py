import torch

#dataloader不要加import

def pre_sample(dataloader, num_epoch:int,node_size:int,edge_size:int):
    nodes_counts = torch.zeros(dataloader.graph.num_nodes,dtype = torch.long)
    edges_counts = torch.zeros(dataloader.graph.num_edges,dtype = torch.long)
    print(nodes_counts.shape,edges_counts.shape)
    sampler = dataloader.sampler
    neg_sampling = dataloader.neg_sampler
    sample_fn = dataloader.sampler_fn
    graph = dataloader.graph
    for _ in range(num_epoch):
        dataloader.__iter__()
        while dataloader.recv_idxs < dataloader.expected_idx:
            dataloader.recv_idxs += 1
            data = dataloader._next_data()
            out = sample_fn(sampler,data,neg_sampling)
            if(len(out)>0):
                sample_out,metadata = out     
            else:
                sample_out = out
            eid = [ret.eid() for ret in sample_out]
            eid_tensor = torch.cat(eid,dim = 0)
            src_node = graph.sample_graph['edge_index'][0,eid_tensor*2].to(graph.nids_mapper.device)
            dst_node = graph.sample_graph['edge_index'][1,eid_tensor*2].to(graph.nids_mapper.device)
            eid_tensor = torch.unique(eid_tensor)
            nid_tensor = torch.unique(torch.cat((src_node,dst_node)))
            edges_counts[eid_tensor] += 1
            nodes_counts[nid_tensor] += 1
    return nodes_counts,edges_counts

