import argparse
import os
import sys
from os.path import abspath, join, dirname
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex
from starrygl.module.modules import GeneralModel

from starrygl.module.utils import parse_config
from starrygl.sample.graph_core import DataSet, GraphData, TemporalNeighborSampleGraph
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
"""
test command 
python test.py --world_size 2 --rank 0 
--world_size', default=4, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='rank of the worker')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed  for reproducibility')
parser.add_argument('--num_sampler', type=int, default=10, metavar='S',
                    help='number of samplers')
parser.add_argument('--queue_size', type=int, default=10, metavar='S',
                    help='sampler queue size')
"""
parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--rank', default=0, type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--world_size', default=1, type=int, metavar='W',
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
    pdata = partition_load("./dataset/here/WIKI", algo="metis_for_tgnn")    
    graph = GraphData(pdata = pdata)
    #dist.barrier()  
    #for i in range(100):
    #    print(i)
    dist.barrier()
    idx = ((graph.eids_mapper >> 48).int() & 0xFFFF)
    print((idx==0).nonzero().shape,(idx==1).nonzero().shape)
    t1 = time.time()
    """
    fut = []
    for i in range(1000):
        #print(i)
        out = graph.edge_attr.index_select(graph.eids_mapper[(idx== 0)|(idx ==1)].to('cuda'))
        fut.append(out)
        #out.wait()
        #out.value()
        if i>0 and i%100==0:
            f = torch.futures.collect_all(fut)
            f.wait()
            f.value()
            fut = []
    """
    partptr  = torch.tensor([ ((i & 0xFFFF)<<48) for i in range(3) ],device = 'cuda')
    
    for i in range(1000):
        if i%100==0:
            idx = graph.eids_mapper.to('cuda')
            idx,inv = idx.unique(return_inverse=True)
            ind = torch.searchsorted(idx,partptr,right=False)
            len = ind[1:]-ind[:-1]
            gatherlen = torch.empty([2],dtype = torch.long,device = 'cuda')
            dist.all_to_all_single(gatherlen,len)
            query_idx = torch.empty([gatherlen.sum()],dtype = torch.long,device = 'cuda')
            input_s = list(len)
            output_s = list(gatherlen)
            dist.all_to_all_single(query_idx,idx,output_s,input_s)
            input_f = graph.edge_attr.accessor.data[DistIndex(query_idx).loc]
        f = torch.empty([idx.shape[0],graph.edge_attr.accessor.data.shape[1]],dtype=torch.float,device='cuda')
        dist.all_to_all_single(f,input_f,input_s,output_s)
    
    torch.cuda.synchronize()
    t2 = time.time()-t1
    print(t2)
    #dist.barrier()
    ctx.shutdown()
if __name__ == "__main__":
    main()

