Distributed Training
====================

Preparation For Distributed Environment
---------------------------------------

Before start training, we need to prepare the environment for distributed training, inluding the following steps:

1. Initialize the Distributed context

    .. code-block:: python

        ctx = DistributedContext.init(backend="nccl", use_gpu=True) 

2. Load the partitioned dataset

    .. code-block:: python

        pdata = partition_load("/mnt/data/part_data/dataset/here/{}".format(args.dataname), algo="metis_for_tgnn")    
        graph = DistributedGraphStore(pdata = pdata)
        sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
        train_data = torch.masked_select(graph.edge_index,pdata.train_mask.to(graph.edge_index.device)).reshape(2,-1)
        train_ts = torch.masked_select(graph.edge_ts,pdata.train_mask.to(graph.edge_index.device))
        val_data = torch.masked_select(graph.edge_index,pdata.val_mask.to(graph.edge_index.device)).reshape(2,-1)
        val_ts = torch.masked_select(graph.edge_ts,pdata.val_mask.to(graph.edge_index.device))
        test_data = torch.masked_select(graph.edge_index,pdata.test_mask.to(graph.edge_index.device)).reshape(2,-1)
        test_ts = torch.masked_select(graph.edge_ts,pdata.test_mask.to(graph.edge_index.device)) 
        train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(pdata.train_mask).view(-1))
        test_data = DataSet(edges = test_data,ts =test_ts,eids = torch.nonzero(pdata.test_mask).view(-1))
        val_data = DataSet(edges = val_data,ts = val_ts,eids = torch.nonzero(pdata.val_mask).view(-1))
        train_stream = torch.cuda.Stream()
        send_stream = torch.cuda.Stream()
        scatter_stream = torch.cuda.Stream()

3. Construct Mailbox and sampler

    .. code-block:: python

        mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat = pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0)
        sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=num_layers, fanout=fanout,graph_data=sample_graph, workers=10,policy = policy, graph_name = "wiki_train")
        neg_sampler = NegativeSampling('triplet')

4. Construct the DataLoader

    .. code-block:: python

        trainloader = DistributedDataLoader(graph,train_data,sampler = sampler,
                                            sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                            neg_sampler=neg_sampler,
                                            batch_size = train_param['batch_size'],
                                            shuffle=False,
                                            drop_last=True,
                                            chunk_size = None,
                                            train=True,
                                            queue_size = 1000,
                                            mailbox = mailbox)
        testloader = DistributedDataLoader(graph,test_data,sampler = sampler,
                                            sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                            neg_sampler=neg_sampler,
                                            batch_size = train_param['batch_size'],
                                            shuffle=False,
                                            drop_last=False,
                                            chunk_size = None,
                                            train=False,
                                            queue_size = 100,
                                            mailbox = mailbox)
        valloader = DistributedDataLoader(graph,val_data,sampler = sampler,
                                            sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                            neg_sampler=neg_sampler,
                                            batch_size = train_param['batch_size'],
                                            shuffle=False,
                                            drop_last=False,
                                            chunk_size = None,
                                            train=False,
                                            queue_size = 100,
                                            mailbox = mailbox)

5. `Create the Model <module.rst>`_

6. Construct the optimizer, early stopper and creterion

    .. code-block:: python

        creterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
        early_stopper = EarlyStopMonitor(max_round=args.patience)

7. Start Training

    .. code-block:: python

         for e in range(train_param['epoch']):
            torch.cuda.synchronize()
            model.train()
            if mailbox is not None:
                mailbox.reset()
                model.module.memory_updater.last_updated_nid = None
                model.module.memory_updater.last_updated_memory = None
                model.module.memory_updater.last_updated_ts = None
            for roots,mfgs,metadata,sample_time in trainloader:
                with torch.cuda.stream(train_stream):
                    optimizer.zero_grad()
                    pred_pos, pred_neg = model(mfgs,metadata)
                    loss = creterion(pred_pos, torch.ones_like(pred_pos))
                    loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                    total_loss += float(loss)
                    loss.backward()
                    optimizer.step()
                    y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                    y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                    train_aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
                    if mailbox is not None:
                        src = metadata['src_pos_index']
                        dst = metadata['dst_pos_index']
                        ts = roots.ts
                        if graph.edge_attr is None:
                            edge_feats = None
                        elif(graph.edge_attr.device == torch.device('cpu')):
                            edge_feats = graph.edge_attr[roots.eids.to('cpu')].to('cuda')
                        else:
                            edge_feats = graph.edge_attr[roots.eids] 
                        dist_index_mapper = mfgs[0][0].srcdata['ID']
                        root_index = torch.cat((src,dst))
                        last_updated_nid = model.module.memory_updater.last_updated_nid[root_index]
                        last_updated_memory = model.module.memory_updater.last_updated_memory[root_index]
                        last_updated_ts=model.module.memory_updater.last_updated_ts[root_index]
                        index, memory, memory_ts = mailbox.get_update_memory(last_updated_nid,
                                                                        last_updated_memory,
                                                                        last_updated_ts)
                        index, mail, mail_ts = mailbox.get_update_mail(dist_index_mapper,
                                                    src,dst,ts,edge_feats,
                                                    model.module.memory_updater.last_updated_memory, 
                                                    model.module.embedding,use_src_emb,use_dst_emb,
                                                    )
                        mailbox.set_mailbox_all_to_all(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max')
            train_ap = float(torch.tensor(train_aps).mean())    
            ap, auc = eval('val')
            print('\ttrain loss:{:.4f}  train ap:{:4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss,train_ap, ap, auc))

8. Deifine the Evaluation function
   
   .. code-block:: python

        def eval(mode='val'):
            model.eval()
            aps = list()
            aucs_mrrs = list()
            if mode == 'val':
                loader = valloader
            elif mode == 'test':
                loader = testloader
            elif mode == 'train':
                loader = trainloader
            with torch.no_grad():
                total_loss = 0
                for roots,mfgs,metadata,sample_time in loader:
                    
                    pred_pos, pred_neg = model(mfgs,metadata)
                    total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
                    total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                    y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                    y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                    aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
                    aucs_mrrs.append(roc_auc_score(y_true, y_pred))
                    if mailbox is not None:
                        src = metadata['src_pos_index']
                        dst = metadata['dst_pos_index']
                        
                        ts = roots.ts
                        if graph.edge_attr is None:
                            edge_feats = None
                        elif(graph.edge_attr.device == torch.device('cpu')):
                            edge_feats = graph.edge_attr[roots.eids.to('cpu')].to('cuda')
                        else:
                            edge_feats = graph.edge_attr[roots.eids] 
                        dist_index_mapper = mfgs[0][0].srcdata['ID']
                        root_index = torch.cat((src,dst))
                        last_updated_nid = model.module.memory_updater.last_updated_nid[root_index]
                        last_updated_memory = model.module.memory_updater.last_updated_memory[root_index]
                        last_updated_ts=model.module.memory_updater.last_updated_ts[root_index]
                        index, memory, memory_ts = mailbox.get_update_memory(last_updated_nid,
                                                                        last_updated_memory,
                                                                        last_updated_ts)
                        
                        index, mail, mail_ts = mailbox.get_update_mail(dist_index_mapper,
                                                    src,dst,ts,edge_feats,
                                                    model.module.memory_updater.last_updated_memory,
                                                    model.module.embedding,use_src_emb,use_dst_emb,
                                                    )
                        mailbox.set_mailbox_all_to_all(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max')

9. Start Evaluation

    .. code-block::python

        if mailbox is not None:
            mailbox.reset()
            model.module.memory_updater.last_updated_nid = None
            print("Train eval:", eval('train'))
            print("Val eval:", eval('test'))
        ap, auc = eval('val')
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))



