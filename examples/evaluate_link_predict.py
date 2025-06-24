import argparse
import os
import sys
from os.path import abspath, join, dirname
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex
from starrygl.evaluation.get_evalute_data import get_link_prediction_data
from starrygl.module.modules import GeneralModel
from pathlib import Path

from pathlib import Path


from starrygl.module.utils import parse_config
from starrygl.sample.cache.fetch_cache import FetchFeatureCache
from starrygl.sample.graph_core import DataSet, DistributedGraphStore, TemporalNeighborSampleGraph
from starrygl.module.utils import parse_config, EarlyStopMonitor
from starrygl.sample.graph_core import DataSet, DistributedGraphStore, TemporalNeighborSampleGraph
from starrygl.sample.memory.shared_mailbox import SharedMailBox
from starrygl.sample.sample_core.EvaluateNegativeSampling import EvaluateNegativeSampling

from starrygl.sample.sample_core.base import NegativeSampling
from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler
from starrygl.sample.part_utils.partition_tgnn import partition_load
import torch
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os

from starrygl.sample.data_loader import DistributedDataLoader
from starrygl.sample.batch_data import SAMPLE_TYPE
from starrygl.sample.stream_manager import getPipelineManger

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='name of dataset')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--world_size', default=1, type=int, metavar='W',
                    help='number of negative samples')
parser.add_argument('--dataname', default=1, type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--model', default='TGN', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--negative_sample_strategy', default='random', type=str, metavar='W',
                    help='name of negative sample strategy')
parser.add_argument('--negative_sample_strategy', default='random', type=str, metavar='W',
                    help='name of negative sample strategy')
args = parser.parse_args()
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import time
import random
import dgl
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)
#os.environ["RANK"] = str(args.rank)
#os.environ["WORLD_SIZE"] = str(args.world_size)
#os.environ["LOCAL_RANK"] = str(0)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
os.environ["MASTER_ADDR"] = '10.214.211.187'
os.environ["MASTER_PORT"] = '9337'
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1234)
def main():   
    print('main')
    use_cuda = True
    sample_param, memory_param, gnn_param, train_param = parse_config('./config/{}.yml'.format(args.model))
    torch.set_num_threads(12)
    ctx = DistributedContext.init(backend="nccl", use_gpu=True)
    device_id = torch.cuda.current_device()
    pdata = partition_load("/mnt/data/part_data/evaluate/{}".format(args.dataname), algo="metis_for_tgnn")    
    graph = DistributedGraphStore(pdata = pdata,uvm_edge = False)
    gnn_param['dyrep'] = True if args.model == 'DyRep' else False
    use_src_emb = gnn_param['use_src_emb'] if 'use_src_emb' in gnn_param else False
    use_dst_emb = gnn_param['use_dst_emb'] if 'use_dst_emb' in gnn_param else False
    gnn_dim_node = 0 if graph.x is None else pdata.x.shape[1]
    gnn_dim_edge = 0 if graph.edge_attr is None else pdata.edge_attr.shape[1]
    print(gnn_dim_node,gnn_dim_edge)
    avg_time  = 0
    MODEL_SAVE_PATH = f'./saved_models/{args.model}-{args.dataname}.pth'
    if use_cuda:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param).cuda()
        device = torch.device('cuda')
    else:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param)
        device = torch.device('cpu')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
    if memory_param['type'] != 'none':
        mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat = pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0)
    else:
        mailbox = None
    fanout = []
    num_layers = sample_param['layer'] if 'layer' in sample_param else 1
    fanout = sample_param['neighbor'] if 'neighbor' in sample_param else [10]
    policy = sample_param['strategy'] if 'strategy' in sample_param else 'recent'
    sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=num_layers, fanout=fanout,graph_data=sample_graph, workers=10,policy = policy, graph_name = "wiki_train")
    train_data = torch.masked_select(graph.edge_index,pdata.train_mask.to(device)).reshape(2,-1)
    train_ts = torch.masked_select(graph.edge_ts,pdata.train_mask.to(device))
    val_data = torch.masked_select(graph.edge_index,pdata.val_mask.to(device)).reshape(2,-1)
    val_ts = torch.masked_select(graph.edge_ts,pdata.val_mask.to(device))
    test_data = torch.masked_select(graph.edge_index,pdata.test_mask.to(device)).reshape(2,-1)
    test_ts = torch.masked_select(graph.edge_ts,pdata.test_mask.to(device)) 
    #print(train_data.shape[1],val_data.shape[1],test_data.shape[1])
    train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(pdata.train_mask).view(-1))
    test_data = DataSet(edges = test_data,ts =test_ts,eids = torch.nonzero(pdata.test_mask).view(-1))
    val_data = DataSet(edges = val_data,ts = val_ts,eids = torch.nonzero(pdata.val_mask).view(-1))
    new_node_val_data = torch.masked_select(graph.edge_index,pdata.new_node_val_mask.to(device)).reshape(2,-1)
    new_node_val_ts = torch.masked_select(graph.edge_ts,pdata.new_node_val_mask.to(device))
    new_node_test_data = torch.masked_select(graph.edge_index,pdata.new_node_test_mask.to(device)).reshape(2,-1)
    new_node_test_ts = torch.masked_select(graph.edge_ts,pdata.new_node_test_mask.to(device)) 
    new_node_val_data = DataSet(edges = new_node_val_data, ts = new_node_val_ts, edis = torch.nonzero(pdata.new_node_val_mask).view(-1))
    new_node_test_data = DataSet(edges = new_node_test_data, ts = new_node_test_ts, edis = torch.nonzero(pdata.new_node_test_mask).view(-1))
    if args.negative_sample_strategy != 'random':
        val_neg_edge_sampler = EvaluateNegativeSampling(src_node_ids=graph.edge_index[0,:], dst_node_ids=graph.edge_index[1,:],
                                                   interact_times=graph.edge_ts, last_observed_time=train_data.ts[-1],
                                                   negative_sample_strategy=args.negative_sample_strategy, seed=0)
        new_node_val_neg_edge_sampler = EvaluateNegativeSampling(src_node_ids=new_node_val_data.edges[0,:], dst_node_ids=new_node_val_data.edges[1,:],
                                                            interact_times=new_node_val_data.ts, last_observed_time=train_data.ts[-1],
                                                            negative_sample_strategy=args.negative_sample_strategy, seed=1)
        test_neg_edge_sampler = EvaluateNegativeSampling(src_node_ids=graph.edge_index[0,:], dst_node_ids=graph.edge_index[1,:],
                                                    interact_times=graph.edge_ts, last_observed_time=val_data.ts[-1],
                                                    negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = EvaluateNegativeSampling(src_node_ids=new_node_test_data.edges[0,:], dst_node_ids=new_node_test_data.edges[1,:],
                                                             interact_times=new_node_test_data.ts, last_observed_time=val_data.ts[-1],
                                                             negative_sample_strategy=args.negative_sample_strategy, seed=3)
    else:
        val_neg_edge_sampler = EvaluateNegativeSampling(src_node_ids=graph.edge_index[0,:], dst_node_ids=graph.edge_index[1,:], seed=0)
        new_node_val_neg_edge_sampler = EvaluateNegativeSampling(src_node_ids=new_node_val_data.edges[0,:], dst_node_ids=new_node_val_data.edges[1,:], seed=1)
        test_neg_edge_sampler =EvaluateNegativeSampling(src_node_ids=graph.edge_index[0,:], dst_node_ids=graph.edge_index[1,:], seed=2)
        new_node_test_neg_edge_sampler = EvaluateNegativeSampling(src_node_ids=new_node_test_data.edges[0,:], dst_node_ids=new_node_test_data.edges[1,:], seed=3)

    
