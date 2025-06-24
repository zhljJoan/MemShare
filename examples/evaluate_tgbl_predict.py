
import argparse
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)
from os.path import abspath, join, dirname
from starrygl.evaluation.evaluator import Evaluator

from starrygl.sample.sample_core.EvaluateNegativeSampling import TgbNegativeSampling
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex
from starrygl.module.modules import GeneralModel
from pathlib import Path

from pathlib import Path


from starrygl.module.utils import parse_config
from starrygl.sample.cache.fetch_cache import FetchFeatureCache
from starrygl.sample.graph_core import DataSet, DistributedGraphStore, TemporalNeighborSampleGraph
from starrygl.module.utils import parse_config, EarlyStopMonitor
from starrygl.sample.graph_core import DataSet, DistributedGraphStore, TemporalNeighborSampleGraph
from starrygl.sample.memory.shared_mailbox import SharedMailBox
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
from starrygl.sample.sample_core.LocalNegSampling import LocalNegativeSampling

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='name of dataset')
parser.add_argument('--local_rank', default=0, type=int, metavar='W',
                    help='name of dataset')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--world_size', default=1, type=int, metavar='W',
                    help='number of negative samples')
parser.add_argument('--dataname', default="WIKI", type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--model', default='TGN', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--num_neg_sampler', default=999, type=int, metavar='W',
                    help='name of negative sample strategy')
parser.add_argument('--inductive', default=True, type=bool, metavar='W',
                    help='name of negative sample strategy')
parser.add_argument('--train_world_size', default=1, type=int, metavar='W',
                    help='name of negative sample strategy')
args = parser.parse_args()
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import time
import random
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)
if not 'WORLD_SIZE'  in os.environ:
    os.environ["RANK"] = str(0)#str(args.rank)
    os.environ["WORLD_SIZE"] = str(1)#str(args.world_size)
    os.environ["LOCAL_RANK"] = str(0)#str(args.local_rank)
if not 'MASTER_ADDR' in os.environ:
    os.environ["MASTER_ADDR"] = '192.168.2.107'
if not 'MASTER_PORT' in os.environ:
    os.environ["MASTER_PORT"] = '9337'
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(34)
def main():   
    print('LOCAL RANK {}, RANK{}'.format(os.environ["LOCAL_RANK"],os.environ["RANK"]))
    use_cuda = True
    sample_param, memory_param, gnn_param, train_param = parse_config('../config/{}.yml'.format(args.model))
    ctx = DistributedContext.init(backend="nccl", use_gpu=True)
    print('initlize distributed')
    torch.set_num_threads(10)#int(/torch.distributed.get_world_size()))
    device_id = torch.cuda.current_device()
    print('use cuda on',device_id)
    pdata = partition_load('/mnt/data/part_data/evaluate/tgbl/{}'.format(args.dataname), algo="metis_for_tgnn")    
    graph = DistributedGraphStore(pdata = pdata)
    
    print(graph.num_nodes)
    Path("../saved_models/").mkdir(parents=True, exist_ok=True)
    Path("../saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    get_checkpoint_path = lambda \
    epoch: f'../saved_checkpoints/{args.model}-{args.dataname}-{epoch}.pth'
    MODEL_SAVE_PATH = f'../saved_models/{args.model}-{args.dataname}-{args.train_world_size}.pth'
    gnn_param['dyrep'] = True if args.model == 'DyRep' else False
    use_src_emb = gnn_param['use_src_emb'] if 'use_src_emb' in gnn_param else False
    use_dst_emb = gnn_param['use_dst_emb'] if 'use_dst_emb' in gnn_param else False
    sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
    if memory_param['type'] != 'none':
        mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat = pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0,ts_dtye = graph.edge_ts.dtype)
    else:
        mailbox = None
    fanout = []
    num_layers = sample_param['layer'] if 'layer' in sample_param else 1
    fanout = sample_param['neighbor'] if 'neighbor' in sample_param else [10]
    policy = sample_param['strategy'] if 'strategy' in sample_param else 'recent'
    sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=num_layers, fanout=fanout,graph_data=sample_graph, workers=10,policy = policy, graph_name = "wiki_train")
    train_data = torch.masked_select(graph.edge_index,pdata.train_mask.to(graph.edge_index.device)).reshape(2,-1)
    train_ts = torch.masked_select(graph.edge_ts,pdata.train_mask.to(graph.edge_index.device))
    test_data = torch.masked_select(graph.edge_index,pdata.test_mask.to(graph.edge_index.device)).reshape(2,-1)
    test_ts = torch.masked_select(graph.edge_ts,pdata.test_mask.to(graph.edge_index.device))
    val_data = torch.masked_select(graph.edge_index,pdata.val_mask.to(graph.edge_index.device)).reshape(2,-1)
    val_ts = torch.masked_select(graph.edge_ts,pdata.val_mask.to(graph.edge_index.device)) 
    train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(pdata.train_mask).view(-1))
    test_data = DataSet(edges = test_data,ts =test_ts,eids = torch.nonzero(pdata.test_mask).view(-1))
    val_data = DataSet(edges = val_data,ts = val_ts,eids = torch.nonzero(pdata.val_mask).view(-1))
    dataset =  PyGLinkPropPredDataset(name=args.dataname, root="datasets")
    print(graph.edge_ts.to(torch.long),dataset.get_TemporalData().t)
    #neg_sampler = LocalNegativeSampling('triplet',args.num_neg_sampler,dst_node_list = graph.edge_index[1,:])
    neg_sampler = TgbNegativeSampling('tgbtriplet',args.num_neg_sampler,dataset.negative_sampler,dataset.dataset.meta_dict)
    train_sampler = NegativeSampling('triplet')
    trainloader = DistributedDataLoader(graph,train_data,sampler = sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=train_sampler,
                                        batch_size = train_param['batch_size'],
                                        shuffle=False,
                                        drop_last=True,
                                        chunk_size = None,
                                        train=True,
                                        queue_size = 200,
                                        mailbox = mailbox,
                                        )
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
    
    print(val_data.ts.to(torch.long)[:100],test_data.ts,train_data.ts)
    gnn_dim_node = 0 if graph.x is None else pdata.x.shape[1]
    gnn_dim_edge = 0 if graph.edge_attr is None else pdata.edge_attr.shape[1]
    avg_time  = 0
    if use_cuda:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param).cuda()
        device = torch.device('cuda')
    else:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param)
        device = torch.device('cpu')
    model = DDP(model,find_unused_parameters=True)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    train_stream = torch.cuda.Stream()
    evaluator = Evaluator()
    def eval(mode='val'):
        model.eval()
        metric = list()
        if mode == 'val':
            loader = valloader
            if isinstance(neg_sampler,TgbNegativeSampling):
                neg_sampler.load_ns(mode)
        elif mode == 'train':
            loader = trainloader
            neg_num = 1
        elif mode == 'test':
            loader = testloader
            if isinstance(neg_sampler,TgbNegativeSampling):
                neg_sampler.load_ns(mode)
        with torch.no_grad():
            total_loss = 0
            signal = torch.tensor([0],dtype = int,device = device)
            with torch.cuda.stream(train_stream):
                for roots,mfgs,metadata in loader:
                    if mode == 'train':
                        neg_num = 1
                    else:
                        neg_num = neg_sampler.amount
                    pred_pos, pred_neg = model(mfgs,metadata,neg_num,'triplet')
                    
                    if mode != 'train':
                        #print(pred_pos,pred_neg)
                        eval_metric = evaluator.eval(pred_pos,pred_neg)
                        metric.append(list(eval_metric.values()))
                        #print(eval_metric)
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
                                                     model.module.embedding,use_src_emb,
                                                     use_dst_emb,
                                                )

                        mailbox.set_mailbox_all_to_all(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max')
        if mode != 'train':
            print('{} metric\n'.format(mode))
            metric = torch.tensor(metric)
            metric_val = metric.mean(dim = 0)
            for i,key in enumerate(evaluator.valid_metric_list):
                print('metric {} is {}\n'.format(key,metric_val[i]))
            print('\n')

    model.eval()
    if mailbox is not None and args.inductive is True:
        mailbox.reset()
        model.module.memory_updater.last_updated_nid = None
        eval('train')
        eval('val')
    eval('test')

    print('test_dataset {} avg_time {} \n'.format(test_data.edges.shape[1],avg_time/train_param['epoch']))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    ctx.shutdown()
if __name__ == "__main__":
    main()

