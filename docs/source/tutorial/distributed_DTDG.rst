Distributed Training
====================

Preparation For Distributed Environment
---------------------------------------

Before start training, we need to prepare the environment for distributed training, including the following steps:

1. Initialize the Distributed context

    .. code-block:: python

        ctx = DistributedContext.init(backend="nccl", use_gpu=True)
        group = ctx.get_default_group()

2. Import the partitioned dataset using the wrapped function, and let the main process (ctx.rank=0) do the data preparation

    .. code-block:: python

        data_root = "./dataset"
        dataset = build_dataset(args)
        if ctx.rank == 0:
            graph, dataset = prepare_data(args, data_root, dist.get_world_size(group), dataset)
        dist.barrier()
        g = get_graph(data_root, group).to(ctx.device)

        def prepare_data(root: str, num_parts):
            dataset = TwitterTennisDatasetLoader().get_dataset()

            x = []
            y = []
            edge_index = []
            edge_times = []
            edge_attr = []
            snapshot_count = 0
            for i, data in enumerate(dataset):
                x.append(data.x[:,None,:])
                y.append(data.y[:,None])
                edge_index.append(data.edge_index)
                print(data.edge_index.shape)
                exit(0)
                edge_times.append(torch.full_like(data.edge_index[0], i))
                edge_attr.append(data.edge_attr)
                snapshot_count += 1
            x = torch.cat(x, dim=1)
            y = torch.cat(y, dim=1)
            edge_index = torch.cat(edge_index, dim=1)
            edge_times = torch.cat(edge_times, dim=0)
            edge_attr = torch.cat(edge_attr, dim=0)

            g = GraphData(edge_index, num_nodes=x.size(0))
            g.node()["x"] = x
            g.node()["y"] = y
            g.edge()["time"] = edge_times
            g.edge()["attr"] = edge_attr
            g.meta()["num_nodes"] = x.size(0)
            g.meta()["num_snapshots"] = snapshot_count

            logging.info(f"GraphData.meta().keys(): {g.meta().keys()}")
            logging.info(f"GraphData.node().keys(): {g.node().keys()}")
            logging.info(f"GraphData.edge().keys(): {g.edge().keys()}")

            g.save_partition(root, num_parts, algorithm="random")
            return g

3. Creating a partitioned parallel-based GNN model :code:`sync_gnn`, and create a classifier and a splitter

    .. code-block:: python

        sync_gnn = build_model(args, graph=g, group=group)
        sync_gnn = sync_gnn.to(ctx.device)

        classifier = Classifier(args.hidden_dim, args.hidden_dim)
        classifier = classifier.to(ctx.device)
        spl = splitter(args, min_time, max_time)

4.Start to train our model

    .. code-block:: python

        trainer = Trainer(args, spl, sync_gnn, classifier, dataset, ctx)
        trainer.train()

        class Trainer():
            def __init__(self, args, splitter, gcn, classifier, dataset, ctx):
                self.args = args
                self.splitter = splitter
                self.gcn = gcn
                self.classifier = classifier
                self.comp_loss = nn.BCELoss()
                self.group = self.gcn.group
                self.graph = self.gcn.graph
                self.ctx = ctx

                self.logger = logger.Logger(args, 1)

                self.num_nodes = dataset.num_nodes
                self.data = dataset
                self.time = {'TRAIN': [], 'VALID': [], 'TEST':[]}

                self.init_optimizers(args)

            def init_optimizers(self, args):
                params = self.gcn.parameters()
                self.gcn_opt = torch.optim.Adam(params, lr=args.learning_rate)
                params = self.classifier.parameters()
                self.classifier_opt = torch.optim.Adam(params, lr=args.learning_rate)
                self.gcn_opt.zero_grad()
                self.classifier_opt.zero_grad()

            def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
                torch.save(state, filename)

            def load_checkpoint(self, filename, model):
                if os.path.isfile(filename):
                    print("=> loading checkpoint '{}'".format(filename))
                    checkpoint = torch.load(filename)
                    epoch = checkpoint['epoch']
                    self.gcn.load_state_dict(checkpoint['gcn_dict'])
                    self.classifier.load_state_dict(checkpoint['classifier_dict'])
                    self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
                    self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
                    self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
                    return epoch
                else:
                    self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
                    return 0

            def train(self):
                self.tr_step = 0
                best_eval_valid = 0
                eval_valid = 0
                epochs_without_impr = 0

                for e in range(self.args.num_epochs):
                    eval_train = self.run_epoch(self.splitter.train, e, 'TRAIN', grad=True)
                    if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                        eval_valid = self.run_epoch(self.splitter.dev, e, 'VALID', grad=False)
                        eval_test = self.run_epoch(self.splitter.test, e, 'TEST', grad=False)
                        if eval_valid > best_eval_valid:
                            best_eval_valid = eval_valid
                            best_test = eval_test
                            epochs_without_impr = 0

                for tmp in self.time.keys():
                    self.ctx.sync_print(tmp, np.mean(self.time[tmp]))
                print(eval_test)

            def run_epoch(self, split, epoch, set_name, grad):
                t0 = time.time()
                log_interval = 1
                if set_name == 'TEST':
                    log_interval = 1
                self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

                torch.set_grad_enabled(grad)

                for s in split:
                    hist_snap_ids = s['hist_ts']
                    label_snap_id = s['label_ts']
                    predictions, labels, label_edge = self.predict(hist_snap_ids, label_snap_id, set_name)

                    loss = self.comp_loss(predictions, labels)
                    if set_name == 'TRAIN':
                        loss.backward()

                        all_reduce_gradients(self.gcn)
                        all_reduce_buffers(self.gcn)
                        all_reduce_gradients(self.classifier)
                        all_reduce_buffers(self.classifier)

                        self.gcn_opt.step()
                        self.classifier_opt.step()

                        self.gcn_opt.zero_grad()
                        self.classifier_opt.zero_grad()

                    if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
                        self.logger.log_minibatch(predictions, labels, loss.detach(), adj=label_edge)
                        dist.barrier()
                    else:
                        self.logger.log_minibatch(predictions, labels, loss.detach())

                torch.set_grad_enabled(True)
                eval_measure = self.logger.log_epoch_done()
                t1 = time.time()
                self.time[set_name].append(t1-t0)

                return eval_measure

            def predict(self, hist_snap_ids, label_snap_id, set_name):
                nodes_embs_dst = self.gcn(hist_snap_ids)
                num_dst = nodes_embs_dst.shape[0]
                nodes_embs_src = self.gcn.route.apply(nodes_embs_dst)
                num_src = nodes_embs_src.shape[0]
                num_nodes, x, pos_edge_index, edge_attr = self.gcn.get_snapshot(label_snap_id)
                neg_edge_index = self.negative_sampling(num_src, num_dst, edge_attr.shape[0], set_name)

                pos_cls_input = self.gather_node_embs(nodes_embs_src, pos_edge_index, nodes_embs_dst)
                neg_cls_input = self.gather_node_embs(nodes_embs_src, neg_edge_index, nodes_embs_dst)

                pos_predictions = self.classifier(pos_cls_input)
                neg_predictions = self.classifier(neg_cls_input)
                pos_label = torch.ones_like(pos_predictions)
                neg_label = torch.zeros_like(neg_predictions)

                pred = torch.cat([pos_predictions, neg_predictions], dim=0)
                label = torch.cat([pos_label, neg_label], dim=0)

                label_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                return pred.sigmoid(), label, label_edge

            def gather_node_embs(self, nodes_embs_src, node_indices, nodes_embs_dist):
                return torch.cat([nodes_embs_src[node_indices[0,:]], nodes_embs_dist[node_indices[1,:]]], dim=1)

            def optim_step(self, loss):
                self.tr_step += 1
                loss.backward()

                if self.tr_step % self.args.steps_accum_gradients == 0:
                    self.gcn_opt.step()
                    self.classifier_opt.step()

                    self.gcn_opt.zero_grad()
                    self.classifier_opt.zero_grad()

            def negative_sampling(self, num_src, num_dst, num_edge, set_name):
                if set_name == 'TRAIN':
                    num_sample = num_edge * self.args.negative_mult_training
                else:
                    num_sample = num_edge * self.args.negative_mult_test

                src = torch.randint(low=0, high=num_src, size=(num_sample,))
                dst = torch.randint(low=0, high=num_dst, size=(num_sample,))

                return torch.vstack([src, dst]).to(self.ctx.device)
