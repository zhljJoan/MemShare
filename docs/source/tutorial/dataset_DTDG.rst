Preparing the Temporal Graph Dataset for DTDG
====================================

In this tutorial, we will show the preparation process of the temporal graph dataset that can be used by StarryGL.

Processing the raw data
-------------
Take elliptic dataset as an example, the raw data files are as follows:

- `elliptic_txs_features.csv`: the node features of the graph
- `elliptic_txs_edgelist.csv`: the edges of the graph of all the time
- `elliptic_txs_classes.csv`: the class of all the nodes of the graph

To better use this dataset on a discrete-time dynamic graph model, we applied some data processing to it, and end up with 3 more files:

- `elliptic_txs_orig2contiguos.csv`: the mapping relationship between the original node and the successive nodes
- `elliptic_txs_nodetime.csv`: the time stamps of nodes of the graph
- `elliptic_txs_edgelist_timed.csv`: the temporal edges of the graph

This dataset is then called elliptic_temporal.The process of getting the most important file `elliptic_txs_edgelist_timed.csv` is as follows:

.. code-block:: python

    import pandas as pd

    mapping = pd.read_csv('elliptic_txs_orig2contiguos.csv')
    edgelist = pd.read_csv('elliptic_txs_edgelist.csv')
    nodetime = pd.read_csv('elliptic_txs_nodetime.csv')

    temp1 = pd.merge(edgelist, mapping, left_on='txId1', right_on='originalId', how='left')
    temp1.drop(['originalId'], axis=1, inplace=True)
    temp1.drop(['txId1'], axis=1, inplace=True)
    temp1 = temp1[['contiguosId', 'txId2']]
    temp1.columns = ['txId1', 'txId2']

    temp2 = pd.merge(temp1, mapping, left_on='txId2', right_on='originalId', how='left')
    temp2.drop(['originalId'], axis=1, inplace=True)
    temp2.drop(['txId2'], axis=1, inplace=True)
    temp2.columns = ['txId1', 'txId2']

    temp3 = pd.merge(temp2, nodetime, left_on='txId1', right_on='txId', how='left')
    temp3.drop(['txId'], axis=1, inplace=True)
    temp3.columns = ['txId1', 'txId2', 'timestep1']

    edgelist_timed = pd.merge(temp3, nodetime, left_on='txId2', right_on='txId', how='left')
    edgelist_timed.drop(['txId'], axis=1, inplace=True)
    edgelist_timed.columns = ['txId1', 'txId2', 'timestep1', 'timestep2']
    edgelist_timed.drop(['timestep2'], axis=1, inplace=True)
    edgelist_timed.columns = ['txId1', 'txId2', 'timestep']

    edgelist_timed.to_csv('elliptic_txs_edgelist_timed.csv', index=False)

Read raw data and preprocess Data
---------------
After the previous step, we will read in our dataset and use a separate wrapped class to process the corresponding data:

.. code-block:: python

    class Elliptic_Temporal_Dataset():
        def __init__(self, path):
            tar_file = os.path.join(path, 'elliptic.tar.gz')
            tar_archive = tarfile.open(tar_file, 'r:gz')
            self.nodes_labels_times = self.load_node_labels(tar_archive)
            self.edges = self.load_transactions(tar_archive)
            self.nodes, self.nodes_feats = self.load_node_feats(tar_archive)
            self.max_degree = get_max_degs(self)

        def load_node_feats(self, tar_archive):
            data = load_data_from_tar('elliptic_txs_features.csv', tar_archive, starting_line=0)
            nodes = data
            nodes_feats = nodes[:,1:]
            self.num_nodes = len(nodes)
            self.feats_per_node = data.size(1) - 1
            return nodes, nodes_feats.float()

        def load_node_labels(self, tar_archive):
            labels = load_data_from_tar('elliptic_txs_classes.csv', tar_archive, replace_unknow=True).long()
            times = load_data_from_tar('elliptic_txs_nodetime.csv', tar_archive, replace_unknow=True).long()
            lcols = Namespace({'nid': 0, 'label': 1})
            tcols = Namespace({'nid':0, 'time':1})
            nodes_labels_times =[]
            for i in range(len(labels)):
                label = labels[i,[lcols.label]].long()
                if label>=0:
                    nid=labels[i,[lcols.nid]].long()
                    time=times[nid,[tcols.time]].long()
                    nodes_labels_times.append([nid , label, time])
            nodes_labels_times = torch.tensor(nodes_labels_times)

            return nodes_labels_times

        def load_transactions(self, tar_archive):
            data = load_data_from_tar('elliptic_txs_edgelist_timed.csv', tar_archive, type_fn=float, tensor_const=torch.LongTensor)
            tcols = Namespace({'source': 0,
                                 'target': 1,
                                 'time': 2})

            data = torch.cat([data,data[:,[1,0,2]]])
            self.max_time = data[:,tcols.time].max()
            self.min_time = data[:,tcols.time].min()
            return {'idx': data, 'vals': torch.ones(data.size(0))}

We construct a wrapped Elliptic_Temporal_Dataset object to store the data. The data object contains the following attributes:

- `nodes_labels_times`: a tensor that contains the label and time information of each node. Each element is a list containing the ID of the node, a label, and the time.
- `edges`: a dictionary that contains 2 keys: idx and vals. The value of the idx key is a tensor containing the source node, the destination node, and the time, and the value of the vals key is a tensor with 1s.
- `nodes`: a tensor that contains the features of each node. Each element is a list of node ids and attributes.
- `nodes_feats`: a tensor that contains only the features for each node.
- `max_degree`: a tensor that stores the maximum out-degree over all time steps in the dataset.

Generate a graph from graph data
---------------
In order to facilitate further processing later, the corresponding graph is generated from the graph dataset using the encapsulated function:

.. code-block:: python

    graph, dataset = prepare_data2(args, data_root, dist.get_world_size(group), dataset)
    def prepare_data2(args, root, num_partitions, dataset):
        hist_adj_list, hist_ndFeats_list, hist_mask_list, existing_nodes = preprocess(dataset)
        edge_index, edge_attr, edge_times, x, exists = [], [], [], [], []
        num_snapshots = len(hist_adj_list)
        for i in range(num_snapshots):
            edge_index.append(hist_adj_list[i]['idx'].t())
            edge_attr.append(hist_adj_list[i]['vals'])
            edge_times.append(torch.full_like(edge_attr[i], i))

            x.append(make_sparse_tensor(hist_ndFeats_list[i], tensor_type='float',
                                            torch_size=[dataset.num_nodes, dataset.feats_per_node]).to_dense()[:, None, :])

        edge_index = torch.cat(edge_index, dim=1)
        edge_times = torch.cat(edge_times, dim=0)

        x = torch.cat(x, dim=1)
        edge_attr = torch.cat(edge_attr, dim=0).type_as(x)

        g = GraphData(edge_index, num_nodes=x.size(0))
        g.node()["x"] = x
        g.edge()["time"] = edge_times
        g.edge()["attr"] = edge_attr
        g.meta()["num_nodes"] = x.size(0)
        g.meta()["num_snapshots"] = num_snapshots

        g.save_partition(root, num_partitions, algorithm="random")

        return g, dataset

Finally, a GraphData object g is obtained, at the same time it is partitioned and saved.It contains the following attributes:

- `x`: the attributes of the nodes
- `edge_times`: the time steps of the edges
- `edge_attr`: the attributes of the edges
- `num_nodes`: the global number of nodes
- `num_snapshots`: the number of snapshots globally
