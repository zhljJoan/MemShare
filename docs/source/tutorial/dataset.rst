Preparing the Temporal Graph Dataset
====================================

In this tutorial, we will show the preparation process of the temporal graph datase that can be used by StarryGL.

Read Raw Data
-------------

Take Wikipedia dataset as an example, the raw data files are as follows:

- `edges.csv`: the temporal edges of the graph
- `node_features.pt`: the node features of the graph
- `edge_features.pt`: the edge features of the graph

Here is an example to read the raw data files:

.. code-block:: python
    
    data_name = args.data_name
    df = pd.read_csv('raw_data/'+data_name+'/edges.csv')
    if os.path.exists('raw_data/'+data_name+'/node_features.pt'):
        n_feat = torch.load('raw_data/'+data_name+'/node_features.pt')
    else:
        n_feat = None
    if os.path.exists('raw_data/'+data_name+'/edge_features.pt'):
        e_feat = torch.load('raw_data/'+data_name+'/edge_features.pt')
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

Preprocess Data
---------------

After reading the raw data, we need to preprocess the data to get the data format that can be used by StarryGL. The following code shows the preprocessing process:

.. code-block:: python

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
    data = Data() #torch_geometric.data.Data()
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
    sample_graph['train_mask'] = data.train_mask[sample_eid]
    sample_graph['test_mask'] = data.test_mask[sample_eid]
    sample_graph['val_mask'] = data.val_mask[sample_eid]
    data.sample_graph = sample_graph

    data.y = torch.zeros(edge_index.shape[1])
    edge_index_dict = {}
    edge_index_dict['edata'] = data.edge_index
    edge_index_dict['sample_data'] = data.sample_graph['edge_index']
    edge_index_dict['neg_data'] = torch.cat([neg_src.view(1, -1),
                                            dst.view(-1, 1).repeat(1, neg_nums).
                                            reshape(1, -1)], dim=0)
    data.edge_index_dict = edge_index_dict
    edge_weight_dict = {}
    edge_weight_dict['edata'] = 2*neg_nums
    edge_weight_dict['sample_data'] = 1*neg_nums
    edge_weight_dict['neg_data'] = 1

We construct a torch_geometric.data.Data object to store the data. The data object contains the following attributes:

- `num_nodes`: the number of nodes in the graph
- `num_edges`: the number of edges in the graph
- `edge_index`: the edge index of the graph
- `edge_ts`: the timestamp of the edges
- `neg_sample`: the negative samples of the edges
- `x`: the node features of the graph
- `edge_attr`: the edge features of the graph
- `train_mask`: the train mask of the edges
- `val_mask`: the validation mask of the edges
- `test_mask`: the test mask of the edges
- `sample_graph`: the sampled graph
- `edge_index_dict`: the edge index of the sampled graph

Finally, we can partition the graph and save the data:

.. code-block:: python

    partition_save('./dataset/here/'+data_name, data, 16, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
