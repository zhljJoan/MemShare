Creating Temporal GNN Models
============================

Continuous-time Temporal GNN Models
-----------------------------------

To create a continuous-time temporal GNN model, we first need to define a configuration file with the suffix yml to specify the model structures and parameters. Here we use the configuration file :code:`TGN.yml` for TGN model as an example:

.. code-block:: yaml

    sampling:
      - layer: 1
        neighbor: 
          - 10
        strategy: 'recent'
        prop_time: False
        history: 1
        duration: 0
        num_thread: 32
    memory: 
      - type: 'node'
        dim_time: 100
        deliver_to: 'self'
        mail_combine: 'last'
        memory_update: 'gru'
        mailbox_size: 1
        combine_node_feature: True
        dim_out: 100
    gnn:
      - arch: 'transformer_attention'
        use_src_emb: False
        use_dst_emb: False
        layer: 1
        att_head: 2
        dim_time: 100
        dim_out: 100
    train:
      - epoch: 20
        batch_size: 200
        # reorder: 16
        lr: 0.0001
        dropout: 0.2
        att_dropout: 0.2
        all_on_gpu: True

The configuration file is composed of four parts: :code:`sampling`, :code:`memory`, :code:`gnn` and :code:`train`. Here are their meanings:

- :code:`sampling`: This part specifies the sampling strategy for the temporal graph. :code:`layer` field specifies the number of layers in the sampling strategy. The :code:`neighbor` field specifies the number of neighbors to sample for each layer. The :code:`strategy` field specifies the sampling strategy(recent or uniform). The :code:`prop_time` field specifies whether to propagate the time information. The :code:`history` field specifies the number of historical timestamps to use. The :code:`duration` field specifies the duration of the time window. The :code:`num_thread` field specifies the number of threads to use for sampling.
- :code:`memory`: This part specifies the memory module. :code:`type` field specifies the type of memory module(node or none). :code:`dim_time` field specifies the dimension of the time embedding. :code:`deliver_to` field specifies the destination of the message. :code:`mail_combine` field specifies the way to combine the messages. :code:`memory_update` field specifies the way to update the memory. :code:`mailbox_size` field specifies the size of the mailbox. :code:`combine_node_feature` field specifies whether to combine the node features. :code:`dim_out` field specifies the dimension of the output.
- :code:`gnn`: This part specifies the GNN module. :code:`arch` field specifies the architecture of the GNN module. :code:`use_src_emb` field specifies whether to use the source embedding. :code:`use_dst_emb` field specifies whether to use the destination embedding. :code:`layer` field specifies the number of layers in the GNN module. :code:`att_head` field specifies the number of attention heads. :code:`dim_time` field specifies the dimension of the time embedding. :code:`dim_out` field specifies the dimension of the output.
- :code:`train`: This part specifies the training parameters. :code:`epoch` field specifies the number of epochs. :code:`batch_size` field specifies the batch size. :code:`lr` field specifies the learning rate. :code:`dropout` field specifies the dropout rate. :code:`att_dropout` field specifies the attention dropout rate. :code:`all_on_gpu` field specifies whether to put all the data on GPU.

After defining the configuration file, we can firstly read the parameters from the configuration file and create the model by constructing a :code:`General Model` object:

.. code-block:: python

    def parse_config(f):
        conf = yaml.safe_load(open(f, 'r'))
        sample_param = conf['sampling'][0]
        memory_param = conf['memory'][0]
        gnn_param = conf['gnn'][0]
        train_param = conf['train'][0]
        return sample_param, memory_param, gnn_param, train_param
    
    sample_param, memory_param, gnn_param, train_param = parse_config('./config/{}.yml'.format(args.model))
    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param).cuda()
    model = DDP(model)

Then a :code:`GeneralModel` object is created. If needed, we can adjust the model's parameters by modifying the contents of the configuration file. Here we provide 5 models for continuous-time temporal GNNs:

- :code:`TGN`: The TGN model proposed in `Temporal Graph Networks for Deep Learning on Dynamic Graphs <https://arxiv.org/abs/2006.10637>`__.
- :code:`DyRep`: The DyRep model proposed in `Representation Learning and Reasoning on Temporal Knowledge Graphs <https://arxiv.org/abs/2010.02844>`__.
- :code:`TIGER`: The TIGER model proposed in `TIGER: A Transformer-Based Framework for Temporal Knowledge Graph Completion <https://arxiv.org/abs/2101.00529>`__.
- :code:`Jodie`: The Jodie model proposed in `JODIE: Joint Optimization of Dynamics and Importance for Online Embedding <https://arxiv.org/abs/1902.10197>`__.
- :code:`TGAT`: The TGAT model proposed in `Temporal Graph Attention for Deep Temporal Modeling <https://arxiv.org/abs/2002.06902>`__.