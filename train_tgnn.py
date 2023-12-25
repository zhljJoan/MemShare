import argparse
import os
import sys
from os.path import abspath, join, dirname
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex
from starrygl.module.modules import GeneralModel

from starrygl.module.utils import parse_config
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

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--rank', default=0, type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--world_size', default=1, type=int, metavar='W',
                    help='number of negative samples')
parser.add_argument('--dataname', default=1, type=str, metavar='W',
                    help='number of negative samples')
args = parser.parse_args()
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import time
import random
import dgl
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)
os.environ["RANK"] = str(args.rank)
os.environ["WORLD_SIZE"] = str(args.world_size)
os.environ["LOCAL_RANK"] = str(0)
os.environ["MASTER_ADDR"] = '127.0.0.1'
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
    use_cuda = True
    sample_param, memory_param, gnn_param, train_param = parse_config('./config/TGN.yml')
    torch.set_num_threads(12)
    ctx = DistributedContext.init(backend="nccl", use_gpu=True)
    device_id = torch.cuda.current_device()
    print('use cuda on',device_id)
    pdata = partition_load("/mnt/data/part_data/dataset/here/{}".format(args.dataname), algo="metis_for_tgnn")    
    graph = DistributedGraphStore(pdata = pdata)

    sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
    mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat = pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0)
    sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=1, fanout=[10],graph_data=sample_graph, workers=10,policy = 'recent',graph_name = "wiki_train")
    train_data = torch.masked_select(graph.edge_index,pdata.train_mask.to(graph.edge_index.device)).reshape(2,-1)
    train_ts = torch.masked_select(graph.edge_ts,pdata.train_mask.to(graph.edge_index.device))
    val_data = torch.masked_select(graph.edge_index,pdata.val_mask.to(graph.edge_index.device)).reshape(2,-1)
    val_ts = torch.masked_select(graph.edge_ts,pdata.val_mask.to(graph.edge_index.device))
    test_data = torch.masked_select(graph.edge_index,pdata.test_mask.to(graph.edge_index.device)).reshape(2,-1)
    test_ts = torch.masked_select(graph.edge_ts,pdata.test_mask.to(graph.edge_index.device)) 
    #print(train_data.shape[1],val_data.shape[1],test_data.shape[1])
    train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(pdata.train_mask).view(-1))
    test_data = DataSet(edges = test_data,ts =test_ts,eids = torch.nonzero(pdata.test_mask).view(-1))
    val_data = DataSet(edges = val_data,ts = val_ts,eids = torch.nonzero(pdata.val_mask).view(-1))
    #train_neg_sampler = PreNegativeSampling('triplet',torch.masked_select(pdata.edge_index['pos_edge'],graph.data.train_mask).reshape(2,-1))
    neg_sampler = NegativeSampling('triplet')
    trainloader = DistributedDataLoader(graph,train_data,sampler = sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=neg_sampler,
                                        batch_size = 1000,
                                        shuffle=False,
                                        drop_last=True,
                                        chunk_size = None,
                                        train=True,
                                        queue_size = 1000,
                                        mailbox = mailbox)
    testloader = DistributedDataLoader(graph,test_data,sampler = sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=neg_sampler,
                                        batch_size = 1000,
                                        shuffle=False,
                                        drop_last=False,
                                        chunk_size = None,
                                        train=False,
                                        queue_size = 100,
                                        mailbox = mailbox)
    valloader = DistributedDataLoader(graph,val_data,sampler = sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=neg_sampler,
                                        batch_size = 1000,
                                        shuffle=False,
                                        drop_last=False,
                                        chunk_size = None,
                                        train=False,
                                        queue_size = 100,
                                        mailbox = mailbox)
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
    train_stream = torch.cuda.Stream()
    send_stream = torch.cuda.Stream()
    scatter_stream = torch.cuda.Stream()

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
            signal = torch.tensor([0],dtype = int,device = device)
            
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
                                                 )
                    mailbox.set_mailbox_all_to_all(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max')
                

        #ap = float(torch.tensor(aps).mean())
        #if neg_samples > 1:
        #    auc_mrr = float(torch.cat(aucs_mrrs).mean())
        #else:
        #    auc_mrr = float(torch.tensor(aucs_mrrs).mean())
        world_size = dist.get_world_size()
        apc = torch.empty([loader.expected_idx*world_size],dtype = torch.float,device='cuda')
        auc_mrr = torch.empty([loader.expected_idx*world_size],dtype = torch.float,device = 'cuda')
        dist.all_gather_into_tensor(apc,torch.tensor(aps,device ='cuda',dtype=torch.float))
        dist.all_gather_into_tensor(auc_mrr,torch.tensor(aucs_mrrs,device ='cuda',dtype=torch.float))
        ap = float(torch.tensor(apc).mean())
        auc_mrr = float(torch.tensor(auc_mrr).mean())
        return ap, auc_mrr
    
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
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
        if mailbox is not None:
            mailbox.reset()
            model.module.memory_updater.last_updated_nid = None
            model.module.memory_updater.last_updated_memory = None
            model.module.memory_updater.last_updated_ts = None
        for roots,mfgs,metadata,sample_time in trainloader:
            fetch_time +=sample_time/1000
            t_prep_s = time.time()
            
            with torch.cuda.stream(train_stream):

                optimizer.zero_grad()
                pred_pos, pred_neg = model(mfgs,metadata)
                
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                total_loss += float(loss)
                loss.backward()
                optimizer.step()
                #torch.cuda.synchronize()
                t_prep_s = time.time()
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
                                                 )
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    mailbox.set_mailbox_all_to_all(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max')
                    end_event.record()
                    torch.cuda.synchronize()
                    write_back_time += start_event.elapsed_time(end_event)/1000
                
        torch.cuda.synchronize()
        time_prep = time.time() - epoch_start_time
        avg_time += time.time() - epoch_start_time
        train_ap = float(torch.tensor(train_aps).mean())    
        ap = 0
        auc = 0
        ap, auc = eval('val')
        print('\ttrain loss:{:.4f}  train ap:{:4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss,train_ap, ap, auc))
        print('\ttotal time:{:.2f}s  prep time:{:.2f}s'.format(time.time()-epoch_start_time, time_prep))    
        print('\t fetch time:{:.2f}s write back time:{:.2f}s'.format(fetch_time,write_back_time))
          
    model.eval()
    if mailbox is not None:
        mailbox.reset()
        model.module.memory_updater.last_updated_nid = None
        eval('train')
        eval('val')
    ap, auc = eval('test')
    eval_neg_samples = 1
    if eval_neg_samples > 1:
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
    else:
        print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))    
    print('test_dataset',test_data.edges.shape[1],'avg_time',avg_time/train_param['epoch'])
    ctx.shutdown()
if __name__ == "__main__":
    main()

