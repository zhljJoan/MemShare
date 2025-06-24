import argparse
import os
import sys
from os.path import abspath, join, dirname

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
sys.path.append(parent_path)
from pre_train import DotPredictor, TSAGEModel
from starrygl.module.layers import EdgePredictor
from starrygl.sample.sample_core.LocalNegSampling import LocalNegativeSampling
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
parser.add_argument('--model', default='pre', type=str, metavar='W',
                    help='name of model')
args = parser.parse_args()
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import time
import random
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP

#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)
if not 'WORLD_SIZE'  in os.environ:
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
if not 'MASTER_ADDR' in os.environ:
    os.environ["MASTER_ADDR"] = '192.168.2.107'
if not 'MASTER_PORT' in os.environ:
    os.environ["MASTER_PORT"] = '9337'
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
local_rank = os.environ["LOCAL_RANK"]
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
    sample_param, memory_param, gnn_param, train_param = parse_config('../../config/{}.yml'.format(args.model))
    ctx = DistributedContext.init(backend="nccl", use_gpu=True)
    torch.set_num_threads(int(40/torch.distributed.get_world_size()))
    device_id = torch.cuda.current_device()
    print('use cuda on',device_id)
    pdata = partition_load('/mnt/data/part_data/evaluate/tgbl/{}'.format(args.dataname), algo="metis_for_tgnn")    
    #pdata = partition_load("/mnt/data/part_data/v2/here/{}".format(args.dataname), algo="metis_for_tgnn")    
    graph = DistributedGraphStore(pdata = pdata)
    print(graph.num_nodes)
    Path("../../saved_models/").mkdir(parents=True, exist_ok=True)
    Path("../../saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    get_checkpoint_path = lambda \
    epoch: f'../../saved_checkpoints/{args.model}-{args.dataname}-{epoch}.pth'
    gnn_param['dyrep'] = True if args.model == 'DyRep' else False
    use_src_emb = gnn_param['use_src_emb'] if 'use_src_emb' in gnn_param else False
    use_dst_emb = gnn_param['use_dst_emb'] if 'use_dst_emb' in gnn_param else False
    sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
    if memory_param['type'] != 'none':
        mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat = pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0)
    else:
        mailbox = None
    fanout = []
    num_layers = sample_param['layer'] if 'layer' in sample_param else 1
    fanout = sample_param['neighbor'] if 'neighbor' in sample_param else [10]
    policy = sample_param['strategy'] if 'strategy' in sample_param else 'recent'
    sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=num_layers, fanout=fanout,graph_data=sample_graph, workers=int(40/torch.distributed.get_world_size()),policy = policy, graph_name = "wiki_train")
    train_data = torch.masked_select(graph.edge_index,pdata.train_mask.to(graph.edge_index.device)).reshape(2,-1)
    train_ts = torch.masked_select(graph.edge_ts,pdata.train_mask.to(graph.edge_index.device))
    test_data = torch.masked_select(graph.edge_index,pdata.test_mask.to(graph.edge_index.device)).reshape(2,-1)
    test_ts = torch.masked_select(graph.edge_ts,pdata.test_mask.to(graph.edge_index.device))
    val_data = torch.masked_select(graph.edge_index,pdata.val_mask.to(graph.edge_index.device)).reshape(2,-1)
    val_ts = torch.masked_select(graph.edge_ts,pdata.val_mask.to(graph.edge_index.device)) 
    train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(pdata.train_mask).view(-1))
    test_data = DataSet(edges = test_data,ts =test_ts,eids = torch.nonzero(pdata.test_mask).view(-1))
    val_data = DataSet(edges = val_data,ts = val_ts,eids = torch.nonzero(pdata.val_mask).view(-1))
    neg_sampler = LocalNegativeSampling('triplet',dst_node_list = graph.edge_index[1,:].unique())
    trainloader = DistributedDataLoader(graph,train_data,sampler = sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=neg_sampler,
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
    

    gnn_dim_node = 0 if graph.x is None else pdata.x.shape[1]
    gnn_dim_edge = 0 if graph.edge_attr is None else pdata.edge_attr.shape[1]
    avg_time  = 0
    model = TSAGEModel(2,gnn_dim_node,gnn_dim_edge,gnn_param['dim_time'],gnn_param['dim_out'])
    edge_model = EdgePredictor(gnn_param['dim_out'])#DotPredictor()
    device = torch.device('cuda')
    model = DDP(model,find_unused_parameters=True)
    train_stream = torch.cuda.Stream()

    val_losses = list()
    def eval(mode='val'):
        neg_samples = 1
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
            with torch.cuda.stream(train_stream):
                for roots,mfgs,metadata in loader:
                    embedding = model(mfgs)
                    src_embedding = embedding[metadata['src_pos_index']]
                    dst_embedding = embedding[metadata['dst_pos_index']]
                    dst_neg_embedding = embedding[metadata['dst_neg_index']]
                    pred_pos,pred_neg = edge_model(src_embedding,dst_embedding,
                                                   h_neg_dst = dst_neg_embedding,
                    )
                    total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
                    total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                    y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                    y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                    aps.append(average_precision_score(y_true, y_pred.detach().numpy()))

                    aucs_mrrs.append(roc_auc_score(y_true, y_pred))
                

        world_size = dist.get_world_size()
        apc = torch.empty([loader.expected_idx*world_size],dtype = torch.float,device='cuda')
        auc_mrr = torch.empty([loader.expected_idx*world_size],dtype = torch.float,device = 'cuda')
        dist.all_gather_into_tensor(apc,torch.tensor(aps,device ='cuda',dtype=torch.float))
        dist.all_gather_into_tensor(auc_mrr,torch.tensor(aucs_mrrs,device ='cuda',dtype=torch.float))
        ap = float(apc.clone().mean())
        auc_mrr = float(auc_mrr.clone().mean())

        return ap, auc_mrr
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    MODEL_SAVE_PATH = f'../../saved_models/{args.model}-{args.dataname}-{dist.get_world_size()}.pth'
    for e in range(train_param['epoch']):
        torch.cuda.synchronize()
        write_back_time = 0
        fetch_time = 0
        epoch_start_time = time.time()
        train_aps = list()
        print('Epoch {:d}:'.format(e))
        time_prep = 0
        total_loss = 0
        model.train()
        for roots,mfgs,metadata in trainloader:
            #print(e,mfgs)
            #fetch_time +=sample_time/1000
            t_prep_s = time.time()
            
            with torch.cuda.stream(train_stream):

                optimizer.zero_grad()
                embedding = model(mfgs)
                src_embedding = embedding[metadata['src_pos_index']]
                dst_embedding = embedding[metadata['dst_pos_index']]
                dst_neg_embedding = embedding[metadata['dst_neg_index']]
                pred_pos,pred_neg = edge_model(src_embedding,dst_embedding,
                                               h_neg_dst = dst_neg_embedding,
                )
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                total_loss += float(loss)
                loss.backward()
                optimizer.step()
                t_prep_s = time.time()
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                train_aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
        print('val')  
        time_prep = time.time() - epoch_start_time
        avg_time += time.time() - epoch_start_time
        train_ap = float(torch.tensor(train_aps).mean())  

        ap = 0
        auc = 0
        ap, auc = eval('val')
        early_stop = True#early_stopper.early_stop_check(ap)
        if early_stop:
            print("Early stopping at epoch {:d}\n".format(e))
            print(f"Loading the best model at epoch {early_stopper.best_epoch}\n")
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_model_path))
            break
        else:
            print('\ttrain loss:{:.4f}  train ap:{:4f}  val ap:{:4f}  val auc:{:4f}\n'.format(total_loss,train_ap, ap, auc))
            print('\ttotal time:{:.2f}s  prep time:{:.2f}s\n'.format(time.time()-epoch_start_time, time_prep))    
            print('\t fetch time:{:.2f}s write back time:{:.2f}s\n'.format(fetch_time,write_back_time))
            if local_rank == 0:
                torch.save(model.state_dict(), get_checkpoint_path(e))
          
    if not early_stop:        
        print(f"Loading the best model at epoch {early_stopper.best_epoch}")
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        model.load_state_dict(torch.load(best_model_path))  
     
    model.eval()
    ap, auc = eval('test')
    eval_neg_samples = 1
    if eval_neg_samples > 1:
        print('\ttest AP:{:4f}  test MRR:{:4f}\n'.format(ap, auc))
    else:
        print('\ttest AP:{:4f}  test AUC:{:4f}\n'.format(ap, auc))    
    print('test_dataset {} avg_time {} \n'.format(test_data.edges.shape[1],avg_time/train_param['epoch']))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    ctx.shutdown()
if __name__ == "__main__":
    main()

