import argparse
import json
import os
import profile
import sys
import psutil
from os.path import abspath, join, dirname
from torch.cuda.amp import autocast
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)
from starrygl.sample.part_utils.transformer_from_speed import load_from_shared_node_partition, load_from_speed
from starrygl.sample.count_static import time_count
from starrygl.sample.sample_core.LocalNegSampling import LocalNegativeSampling
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex
from starrygl.module.modules import GeneralModel
from pathlib import Path

from pathlib import Path


from starrygl.module.utils import AdaParameter, parse_config
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
from starrygl.sample.count_static import time_count as tt
import os

from starrygl.sample.data_loader import DistributedDataLoader
from starrygl.sample.batch_data import SAMPLE_TYPE
from starrygl.sample.stream_manager import getPipelineManger
from torch.profiler import profile, record_function, ProfilerActivity
parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--rank', default=0, type=int, metavar='W',
                    help='name of dataset')
parser.add_argument('--local_rank', default=0, type=int, metavar='W',
                    help='name of dataset')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--world_size', default=1, type=int, metavar='W',
                    help='number of negative samples')
parser.add_argument('--dataname', default="WIKI", type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--model', default='TGN', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--part_test', default='part', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--partition', default='part', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--topk', default='0', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--probability', default=1, type=float, metavar='W',
                    help='name of model')
parser.add_argument('--sample_type', default='recent', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--local_neg_sample', default=False, type=bool, metavar='W',
                    help='name of model')
parser.add_argument('--shared_memory_ssim', default=2, type=float, metavar='W',
                    help='name of model')
parser.add_argument('--neg_samples', default=1, type=int, metavar='W',
                    help='name of model')
parser.add_argument('--eval_neg_samples', default=1, type=int, metavar='W',
                    help='name of model')
parser.add_argument('--memory_type', default='all_update', type=str, metavar='W',
                    help='name of model')
parser.add_argument('--seed', default=6773, type=int, metavar='W',
                    help='name of model')
parser.add_argument(
    '--coordinated_group_size',
    default=8,
    type=int,
    help='number of root occurrences sharing a keyed random number',
)
parser.add_argument(
    '--max_epoch',
    default=0,
    type=int,
    help='positive value overrides the epoch count in the model config',
)
parser.add_argument(
    '--freeze_adaptive_parameters',
    action='store_true',
    help='hold sampling probability and smooth threshold fixed for pairing',
)
parser.add_argument(
    '--collect_phase_timing',
    action='store_true',
    help='record sampler, AllToAll route/launch, and exposed wait timings',
)
parser.add_argument(
    '--skip_eval',
    action='store_true',
    help='run training epochs without validation and test passes',
)
parser.add_argument(
    '--metrics_path',
    default='',
    type=str,
    help='append rank-aggregated epoch metrics as JSON lines',
)
#boundery_recent_uniform boundery_recent_decay
args = parser.parse_args()
if args.memory_type == 'all_local' or args.topk != '0':
    train_cross_probability = 0
else:
    train_cross_probability = 1
if args.memory_type == 'all_local':
    args.sample_type = 'boundery_recent_uniform'
    args.probability = 0
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import time
import random
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
def get_network_interfaces_with_prefix(prefixs):
    interfaces = psutil.net_if_addrs()
    matching_interfaces = [iface for iface in interfaces if iface.startswith(prefixs[0]) or iface.startswith(prefixs[1])]
    return matching_interfaces

# Example usage
prefix = ("ens4f1np1","ens6f0np0")
matching_interfaces = get_network_interfaces_with_prefix(prefix)
print(f"Network interfaces with prefix '{prefix}': {matching_interfaces}")
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'#str(args.rank)
if not 'WORLD_SIZE'  in os.environ:
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
if not 'MASTER_ADDR' in os.environ:
    os.environ["MASTER_ADDR"] = '192.168.1.107'
if not 'MASTER_PORT' in os.environ:
    os.environ["MASTER_PORT"] = '9337'
    
os.environ["NCCL_IB_DISABLE"]='1'
os.environ['NCCL_SOCKET_IFNAME']=matching_interfaces[0]
print('rank {}'.format(int(os.environ["LOCAL_RANK"])))
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
local_rank = int(os.environ["LOCAL_RANK"])
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(args.seed)
total_next_batch = 0
total_forward = 0
total_count_score = 0
total_backward = 0
total_prepare_mail = 0
total_update_mail = 0
total_update_memory =0
total_remote_update = 0
def count_empty():
    global total_next_batch 
    global total_forward 
    global total_count_score 
    global total_backward 
    global total_prepare_mail 
    global total_update_mail 
    global total_update_memory
    global total_remote_update 
    total_next_batch = 0
    total_forward = 0
    total_count_score = 0
    total_backward = 0
    total_prepare_mail = 0
    total_update_mail = 0
    total_update_memory =0
    total_remote_update = 0
def add(t1,t2,t3,t4,t5,t6,t7,t8):
    global total_next_batch 
    global total_forward 
    global total_count_score 
    global total_backward 
    global total_prepare_mail 
    global total_update_mail 
    global total_update_memory
    global total_remote_update
    total_next_batch += t1
    total_forward += t2
    total_count_score += t4
    total_backward += t3
    total_prepare_mail += t5
    total_update_mail += t6
    total_update_memory +=t7
    total_remote_update += t8
def query():
    global total_next_batch 
    global total_forward 
    global total_count_score 
    global total_backward 
    global total_prepare_mail 
    global total_update_mail 
    global total_update_memory
    global total_remote_update
    global total_next_batch 
    return {
        "total_next_batch":total_next_batch,
        "total_forward" :total_forward ,
    "total_count_score" :total_count_score ,
    "total_backward" :total_backward ,
    "total_prepare_mail" :total_prepare_mail ,
    "total_update_mail" :total_update_mail ,
    "total_update_memory":total_update_memory,
    "total_remote_update":total_remote_update,}


def emit_epoch_metrics(
    ctx,
    epoch,
    train_seconds,
    batch_count,
    batch_latency_samples,
    trainloader,
    local_node,
    remote_node,
    local_edge,
    remote_edge,
    test_ap=None,
    test_auc=None,
    val_ap=None,
    val_auc=None,
):
    """Emit timing records without changing the training execution path."""

    latency = np.asarray(batch_latency_samples, dtype=np.float64)
    local_record = {
        'rank': int(dist.get_rank()),
        'train_seconds': float(train_seconds),
        'batch_count': int(batch_count),
        'batch_latency': {
            'count': int(latency.size),
            'mean_seconds': float(latency.mean()) if latency.size else 0.0,
            'p95_seconds': (
                float(np.percentile(latency, 95)) if latency.size else 0.0
            ),
            'max_seconds': float(latency.max()) if latency.size else 0.0,
        },
        'phase_timing': trainloader.phase_timing_stats(),
    }
    records = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(records, local_record, group=ctx.gloo_group)
    if dist.get_rank() != 0:
        return

    phase_names = sorted(
        {
            key[:-8]
            for record in records
            for key in record['phase_timing']
            if key.endswith('_seconds')
        }
    )
    phases = {}
    for phase in phase_names:
        rank_seconds = [
            float(
                record['phase_timing'].get(
                    '{}_seconds'.format(phase), 0.0
                )
            )
            for record in records
        ]
        calls = sum(
            int(
                record['phase_timing'].get(
                    '{}_calls'.format(phase), 0
                )
            )
            for record in records
        )
        phases[phase] = {
            'calls': calls,
            'critical_rank_seconds': max(rank_seconds, default=0.0),
            'rank_seconds': rank_seconds,
            'total_rank_seconds': sum(rank_seconds),
        }

    report = {
        'dataset': args.dataname,
        'sample_type': args.sample_type,
        'seed': int(args.seed),
        'coordinated_group_size': int(args.coordinated_group_size),
        'world_size': int(dist.get_world_size()),
        'epoch': int(epoch),
        'batch_count_per_rank': int(batch_count),
        'critical_train_seconds': max(
            record['train_seconds'] for record in records
        ),
        'rank_train_seconds': [
            record['train_seconds'] for record in records
        ],
        'max_rank_batch_p95_seconds': max(
            record['batch_latency']['p95_seconds'] for record in records
        ),
        'rank_batch_latency': [
            record['batch_latency'] for record in records
        ],
        'local_nodes': int(local_node.item()),
        'remote_nodes': int(remote_node.item()),
        'local_edges': int(local_edge.item()),
        'remote_edges': int(remote_edge.item()),
        'phase_timing': phases,
        'quality': {
            'val_ap': None if val_ap is None else float(val_ap),
            'val_auc': None if val_auc is None else float(val_auc),
            'test_ap': None if test_ap is None else float(test_ap),
            'test_auc': None if test_auc is None else float(test_auc),
        },
        'instrumentation_note': (
            'phase timers are host-wall measurements of overlapping phases '
            'and must not be summed into epoch latency'
        ),
    }
    encoded = json.dumps(report, sort_keys=True)
    print('COORDINATED_EPOCH_METRICS {}'.format(encoded), flush=True)
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open('a', encoding='utf-8') as output:
            output.write(encoded + '\n')


def main():   
    #torch.backends.cudnn.benchmark = True
    #torch.autograd.set_detect_anomaly(True)
    print('LOCAL RANK {}, RANK{}'.format(os.environ["LOCAL_RANK"],os.environ["RANK"]))
    use_cuda = True
    sample_param, memory_param, gnn_param, train_param = parse_config('../config/{}.yml'.format(args.model))
    memory_param['mode'] = args.memory_type
    ctx = DistributedContext.init(backend="nccl", use_gpu=True,memory_group_num=1,cache_use_rpc=True)
    torch.set_num_threads(10)
    device_id = torch.cuda.current_device()
    if ((args.dataname =='GDELT') & (dist.get_world_size() <=4 )):
        graph,full_sampler_graph,train_mask,val_mask,test_mask,full_train_mask,cache_route = load_from_speed(args.dataname,seed=123457,top=args.topk,sampler_graph_add_rev=True, feature_device=torch.device('cpu'),partition_path=args.partition)#torch.device('cpu'))
    else:
        graph,full_sampler_graph,train_mask,val_mask,test_mask,full_train_mask,cache_route = load_from_speed(args.dataname,seed=123457,top=args.topk,sampler_graph_add_rev=True, feature_device=torch.device('cuda:{}'.format(ctx.local_rank)),partition_path=args.partition)#torch.device('cpu'))
    if args.max_epoch > 0:
        train_param['epoch'] = args.max_epoch
    elif(args.dataname=='GDELT'):
        train_param['epoch'] = 10
    #torch.autograd.set_detect_anomaly(True)
# 确保 CUDA 可用
    if torch.cuda.is_available():
        print("Total GPU memory: ", torch.cuda.get_device_properties(0).total_memory/1024**3)
        print("Current GPU memory allocated: ", torch.cuda.memory_allocated(0)/1024**3)
        print("Current GPU memory reserved: ", torch.cuda.memory_reserved(0)/1024**3)
        print("Max GPU memory allocated during this session: ", torch.cuda.max_memory_allocated(0))
        print("Max GPU memory reserved during this session: ", torch.cuda.max_memory_reserved(0))
    else:
        print("CUDA is not available.")
    
    full_dst = full_sampler_graph['edge_index'][1,torch.arange(0,full_sampler_graph['edge_index'].shape[1],2)]   
    sample_graph = TemporalNeighborSampleGraph(full_sampler_graph,mode = 'full',dist_eid_mapper=graph.eids_mapper)
    eval_sample_graph = TemporalNeighborSampleGraph(full_sampler_graph,mode = 'full',dist_eid_mapper=graph.eids_mapper)
    Path("../saved_models/").mkdir(parents=True, exist_ok=True)
    Path("../saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    get_checkpoint_path = lambda \
    epoch: f'../saved_checkpoints/{args.model}-{args.dataname}-{epoch}.pth'
    gnn_param['dyrep'] = True if args.model == 'DyRep' else False
    use_src_emb = gnn_param['use_src_emb'] if 'use_src_emb' in gnn_param else False
    use_dst_emb = gnn_param['use_dst_emb'] if 'use_dst_emb' in gnn_param else False
    fanout = []
    num_layers = sample_param['layer'] if 'layer' in sample_param else 1
    fanout = sample_param['neighbor'] if 'neighbor' in sample_param else [10]
    policy = sample_param['strategy'] if 'strategy' in sample_param else 'recent'
    no_neg = sample_param['no_neg'] if 'no_neg' in sample_param else False
    print(policy)
    if policy == 'recent':
        policy_train = args.sample_type#'boundery_recent_decay'
        if policy_train == 'recent':
            ada_param = AdaParameter(init_alpha=args.shared_memory_ssim, init_beta=args.probability,min_beta=1)
        else:
            ada_param = AdaParameter(init_alpha=args.shared_memory_ssim, init_beta=args.probability)
    else:
        policy_train = policy
        ada_param = AdaParameter(init_alpha=args.shared_memory_ssim, init_beta=args.probability)
    if memory_param['type'] != 'none':
        mailbox = SharedMailBox(graph.ids.shape[0], memory_param, dim_edge_feat = graph.efeat.shape[1] if graph.efeat is not None else 0,
        shared_nodes_index=graph.shared_nids_list[ctx.memory_group_rank],device = torch.device('cuda:{}'.format(local_rank)),
        start_historical=(args.memory_type=='historical'),
        ada_param = ada_param)
    else:
        mailbox = None

    sampler = NeighborSampler(
        num_nodes=graph.num_nodes,
        num_layers=num_layers,
        fanout=fanout,
        graph_data=sample_graph,
        workers=10,
        policy=policy_train,
        graph_name="train",
        local_part=dist.get_rank(),
        edge_part=DistIndex(graph.eids_mapper).part,
        node_part=DistIndex(graph.nids_mapper).part,
        ada_param=ada_param,
        no_neg=no_neg,
        coordinated_group_size=args.coordinated_group_size,
        coordinated_seed=args.seed,
    )
    eval_sampler = NeighborSampler(
        num_nodes=graph.num_nodes,
        num_layers=num_layers,
        fanout=fanout,
        graph_data=eval_sample_graph,
        workers=10,
        policy=policy_train,
        graph_name="eval",
        local_part=dist.get_rank(),
        edge_part=DistIndex(graph.eids_mapper).part,
        node_part=DistIndex(graph.nids_mapper).part,
        ada_param=ada_param,
        no_neg=no_neg,
        coordinated_group_size=args.coordinated_group_size,
        coordinated_seed=args.seed,
    )

    train_data = torch.masked_select(graph.edge_index,train_mask.to(graph.edge_index.device)).reshape(2,-1)
    train_ts = torch.masked_select(graph.ts,train_mask.to(graph.edge_index.device))
    
    print('part {}\n'.format(DistIndex(graph.nids_mapper[train_data]).part))
    test_range = torch.arange(0,full_sampler_graph['eids'].shape[0],2)
    
    eval_train_data = torch.masked_select(full_sampler_graph['edge_index'][:,test_range],full_train_mask.to(graph.edge_index.device)).reshape(2,-1)
    eval_train_ts = torch.masked_select(full_sampler_graph['ts'][test_range],full_train_mask.to(graph.edge_index.device))
    
    test_data = torch.masked_select(full_sampler_graph['edge_index'][:,test_range],test_mask.to(graph.edge_index.device)).reshape(2,-1)
    test_ts = torch.masked_select(full_sampler_graph['ts'][test_range],test_mask.to(graph.edge_index.device))
    val_data = torch.masked_select(full_sampler_graph['edge_index'][:,test_range],val_mask.to(graph.edge_index.device)).reshape(2,-1)
    val_ts = torch.masked_select(full_sampler_graph['ts'][test_range],val_mask.to(graph.edge_index.device)) 
    train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(train_mask).reshape(-1))
    
    eval_train_data = DataSet(edges = eval_train_data,ts = eval_train_ts,eids = full_train_mask.nonzero().reshape(-1))

    test_data = DataSet(edges = test_data,ts =test_ts,eids = test_mask.nonzero().reshape(-1))
    
    val_data = DataSet(edges = val_data,ts = val_ts,eids = val_mask.nonzero().reshape(-1))
    
    print('ts {} {} {} {}'.format(train_data.ts,eval_train_data.ts,test_data.ts,val_data.ts))
    neg_samples = args.eval_neg_samples
    mask = DistIndex(graph.nids_mapper[graph.edge_index[1,:]].to('cpu')).part == dist.get_rank()
    
    if args.local_neg_sample:
        print('dst len {} origin len {}'.format(graph.edge_index[1,mask].unique().shape[0],full_dst.unique().shape[0]))
        train_neg_sampler = LocalNegativeSampling('triplet',amount = args.neg_samples,dst_node_list = graph.edge_index[1,mask].unique())
        
    else:
        #train_neg_sampler = LocalNegativeSampling('triplet',amount = args.neg_samples,dst_node_list = full_dst.unique())
        train_neg_sampler = LocalNegativeSampling('triplet',amount = args.neg_samples,dst_node_list = full_dst.unique(),local_mask=(DistIndex(graph.nids_mapper[full_dst.unique()].to('cpu')).part == dist.get_rank()),ada_param=ada_param)

    print(train_neg_sampler.dst_node_list)
    neg_sampler = LocalNegativeSampling('triplet',amount= neg_samples,dst_node_list = full_dst.unique(),seed=args.seed)
   
    trainloader = DistributedDataLoader(graph,eval_train_data,sampler = sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=train_neg_sampler,
                                        batch_size = int(train_param['batch_size'])*dist.get_world_size(),
                                        shuffle=False,
                                        drop_last=True,
                                        chunk_size = None,
                                        mode='train',
                                        queue_size = 200,
                                        mailbox = mailbox,
                                        is_pipeline=True,
                                        use_local_feature = False,
                                        device = torch.device('cuda:{}'.format(local_rank)),
                                        probability=args.probability,
                                        reversed = (gnn_param['arch'] == 'identity'),
                                        ada_param=ada_param,
                                        collect_phase_timing=args.collect_phase_timing,
                                        )
    
    eval_trainloader = DistributedDataLoader(graph,eval_train_data,sampler = eval_sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=neg_sampler,
                                        batch_size = train_param['batch_size'],
                                        shuffle=False,
                                        drop_last=False,
                                        chunk_size = None,
                                        mode='eval_train',
                                        queue_size = 100,
                                        mailbox = mailbox,
                                        device = torch.device('cuda:{}'.format(local_rank)),
                                        reversed = (gnn_param['arch']=='identity')
                                        )
    testloader = DistributedDataLoader(graph,test_data,sampler = eval_sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=neg_sampler,
                                        batch_size = train_param['batch_size']*dist.get_world_size(),
                                        shuffle=False,
                                        drop_last=False,
                                        chunk_size = None,
                                        mode='test',
                                        queue_size = 100,
                                        mailbox = mailbox,
                                        device = torch.device('cuda:{}'.format(local_rank)),
                                        reversed = (gnn_param['arch']=='identity')
                                        )
    valloader = DistributedDataLoader(graph,val_data,sampler = eval_sampler,
                                        sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                        neg_sampler=neg_sampler,
                                        batch_size = train_param['batch_size']*dist.get_world_size(),
                                        shuffle=False,
                                        drop_last=False,
                                        chunk_size = None,
                                        train=False,
                                        mode='val',
                                        queue_size = 100,
                                        mailbox = mailbox,
                                        device = torch.device('cuda:{}'.format(local_rank)),
                                        reversed = (gnn_param['arch']=='identity')
                                        )

    print('init dataloader')

    gnn_dim_node = 0 if graph.nfeat is None else graph.nfeat.shape[1]
    gnn_dim_edge = 0 if graph.efeat is None else graph.efeat.shape[1]
    print('dim_node {} dim_edge {}\n'.format(gnn_dim_node,gnn_dim_edge))
    avg_time  = 0
    if use_cuda:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param,graph.ids.shape[0],mailbox,ada_param=ada_param).cuda()
        device = torch.device('cuda')
    else:
        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param,graph.ids.shape[0],mailbox,ada_param=ada_param)
        device = torch.device('cpu')
    model = DDP(model,find_unused_parameters=True)
    def count_parameters(model):
        return sum(p.numel()*p.element_size()/1024/1024 for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    train_stream = torch.cuda.Stream()

    def eval(mode='val'):
        model.eval()
        ada_param.eval()
        aps = list()
        aucs_mrrs = list()
        if mode == 'val':
            loader = valloader
        elif mode == 'test':
            loader = testloader

        elif mode == 'train':
            loader = eval_trainloader
        err_cnt = 0
        err_cross_part = 0
        true_cnt = 0
        true_cross_cnt = 0
        with torch.no_grad():
            total_loss = 0
            signal = torch.tensor([0],dtype = int,device = device)
            batch_cnt = 0
            for roots,mfgs,metadata in loader:
                #print(batch_cnt)
                batch_cnt = batch_cnt+1
                if mailbox is not None:
                    if(graph.efeat.device.type != 'cpu'):
                        edge_feats = graph.get_local_efeat(graph.eids_mapper[roots.eids.to('cpu')]).to('cuda')
                    else:
                        edge_feats = graph.get_local_efeat(graph.eids_mapper[roots.eids.to('cpu')])
                    src = metadata['src_pos_index']
                    dst = metadata['dst_pos_index']
                    ts = roots.ts
                    update_mail = True
                    param = (update_mail,src,dst,ts,edge_feats,loader.async_feature)
                else:
                    param = None
                pred_pos, pred_neg = model(mfgs,metadata,neg_samples=args.neg_samples,async_param = param)
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
                mailbox.update_shared()
                mailbox.update_p2p_mem()
                mailbox.update_p2p_mail()
        ap = torch.empty([1])
        auc_mrr = torch.empty([1])
        #if(ctx.memory_group==0):
        world_size = dist.get_world_size()
        ap[0] = torch.tensor(aps).mean()
        auc_mrr[0] = torch.tensor(aucs_mrrs).mean()#float(aucs_mrrs.clone().mean())
        print('mode: {} {} {}\n'.format(mode,ap,auc_mrr))
        dist.all_reduce(ap,group = ctx.gloo_group)
        ap/=ctx.memory_group_size
        dist.all_reduce(auc_mrr,group=ctx.gloo_group)
        auc_mrr/=ctx.memory_group_size
        dist.broadcast(ap,0,group=ctx.gloo_group)
        dist.broadcast(auc_mrr,0,group=ctx.gloo_group)
        return ap.item(), auc_mrr.item()    
    def normalize(x): 
        if not (x.max().item() == 0):      
            x = x - x.min()
            x = x / x.max() 
            x = 2*x - 1
        return x
    def inner_prod(x1,x2):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(normalize(x1),normalize(x2)).sum()/x1.size(dim=0)
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'],weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    MODEL_SAVE_PATH = f'../saved_models/{args.model}-{args.dataname}-{dist.get_world_size()}.pth'
    total_test_time = 0
    epoch_cnt = 0
    test_ap_list = []
    val_list = []
    loss_list = []
    for e in range(train_param['epoch']):
        ada_param.train()
        model.module.memory_updater.empty_cache()
        trainloader.reset_phase_timing()
        tt._zero()
        torch.cuda.synchronize()
        epoch_start_time = time.time()
        epoch_cnt = epoch_cnt + 1
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
        sum_local_comm = 0
        sum_remote_comm = 0
        sum_local_edge_comm = 0
        sum_remote_edge_comm = 0

        b_cnt = 0
        batch_latency_samples = []
        batch_wall_started = time.perf_counter()
        start = time_count.start_gpu()
        total = 0
        for roots,mfgs,metadata in trainloader:
            end = time_count.elapsed_event(start)
            total += end
            b_cnt = b_cnt + 1

            t1 = time_count.start_gpu()
            if mailbox is not None:
                if(graph.efeat.device.type != 'cpu'):
                    edge_feats = graph.get_local_efeat(graph.eids_mapper[roots.eids.to('cpu')]).to('cuda')#,non_blocking=True)
                else:
                    edge_feats = graph.get_local_efeat(graph.eids_mapper[roots.eids.to('cpu')])
                src = metadata['src_pos_index']
                dst = metadata['dst_pos_index']
                ts = roots.ts
                update_mail = True
                param = (update_mail,src,dst,ts,edge_feats,trainloader.async_feature)
            else:
                param = None
            #print(time_count.elapsed_event(t1))
            model.train()
            t2 = time_count.start_gpu()
            optimizer.zero_grad()
            ones = torch.ones(metadata['dst_neg_index'].shape[0],device = model.device,dtype=torch.float)
            #with autocast():
            pred_pos, pred_neg = model(mfgs,metadata,neg_samples=args.neg_samples,async_param = param)
            #print(time_count.elapsed_event(t2))
            loss = creterion(pred_pos, torch.ones_like(pred_pos)) 
            if args.local_neg_sample is False:
                weight = torch.where(DistIndex(mfgs[0][0].srcdata['ID'][metadata['dst_neg_index']]).part == torch.distributed.get_rank(),ones*metadata['train_ratio_pos'],ones*metadata['train_ratio_neg']).reshape(-1,1)
                neg_creterion = torch.nn.BCEWithLogitsLoss(weight)
                loss += neg_creterion(pred_neg, torch.zeros_like(pred_neg))
            else:
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss.item())
            if len(trainloader.result_queue) > 0:
                _,_,_,edge_feat,node_feat0 = trainloader.result_queue[0]
                if ada_param is not None:
                    ada_param.update_gnn_aggregate_time(ada_param.last_start_event_gnn_aggregate)
                edge_feat[1].wait()
                node_feat0[1].wait()
               
                if (
                    ada_param is not None
                    and e < 3
                    and not args.freeze_adaptive_parameters
                ):
                    
                    ada_param.update_fetch_time(ada_param.last_start_event_fetch)
                    ada_param.update_parameter()
            loss.backward()
            optimizer.step()
            
            
            mailbox.update_shared()
            mailbox.update_p2p_mem()
            mailbox.update_p2p_mail()
            ada_param.last_start_event_memory_update = ada_param.start_event()
            start = time_count.start_gpu()
            batch_wall_finished = time.perf_counter()
            batch_latency_samples.append(
                batch_wall_finished - batch_wall_started
            )
            batch_wall_started = batch_wall_finished
        
        torch.cuda.synchronize()
        dist.barrier()
        time_prep = time.time() - epoch_start_time
        avg_time += time.time() - epoch_start_time
        train_ap = float(torch.tensor(train_aps).mean())  
        print('\ttrain time:{:.2f}s\n'.format(time_prep))    
        print(trainloader.local_node)
        local_node=torch.tensor([trainloader.local_node])
        remote_node=torch.tensor([trainloader.remote_node])
        local_edge=torch.tensor([trainloader.local_edge])
        remote_edge=torch.tensor([trainloader.remote_edge])
        tot_comm_count=torch.tensor([mailbox.tot_comm_count])
        tot_shared_count=torch.tensor([mailbox.tot_shared_count])
        torch.distributed.all_reduce(local_node,group=ctx.gloo_group)
        torch.distributed.all_reduce(remote_node,group=ctx.gloo_group)
        torch.distributed.all_reduce(local_edge,group=ctx.gloo_group)
        torch.distributed.all_reduce(remote_edge,group=ctx.gloo_group)
        torch.distributed.all_reduce(tot_comm_count,group=ctx.gloo_group)
        torch.distributed.all_reduce(tot_shared_count,group=ctx.gloo_group)
        print('local node number {} remote node number {} local edge {} remote edge{}\n'.format(local_node,remote_node,local_edge,remote_edge))
        print(' comm local node number {} remote node number {} local edge {} remote edge{}\n'.format(sum_local_comm,sum_remote_comm,sum_local_edge_comm,sum_remote_edge_comm))
        print('memory comm {} shared comm {}\n'.format(tot_comm_count,tot_shared_count))
        if args.skip_eval:
            emit_epoch_metrics(
                ctx=ctx,
                epoch=e,
                train_seconds=time_prep,
                batch_count=b_cnt,
                batch_latency_samples=batch_latency_samples,
                trainloader=trainloader,
                local_node=local_node,
                remote_node=remote_node,
                local_edge=local_edge,
                remote_edge=remote_edge,
            )
            trainloader.local_node = 0
            trainloader.remote_node = 0
            trainloader.local_edge = 0
            trainloader.remote_edge = 0
            mailbox.tot_comm_count = 0
            mailbox.tot_shared_count = 0
            if e + 1 == train_param['epoch']:
                ctx.shutdown()
                return
            continue
        #if(e==0):
        #    torch.save((local_access,remote_access,local_edge_access,remote_edge_access,local_comm,remote_comm,local_edge_comm,remote_edge_comm),'all_args.seed/{}/{}/comm/comm_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.dataname,args.model,args.partition,args.topk,dist.get_world_size(),dist.get_rank(),args.sample_type,args.probability,args.memory_type,args.shared_memory_ssim))
        ap = 0
        auc = 0
        tt.ssim_remote=0
        tt.ssim_local=0
        tt.weight_count_local=0
        tt.weight_count_remote=0
        tt.ssim_cnt=0
        ap, auc = eval('val')
        print('finish val')
        torch.cuda.synchronize()
        dist.barrier()
        t_test = time.time()
        print('test')
        test_ap,test_auc = eval('test')
        torch.cuda.synchronize()
        dist.barrier()
        t_test = time.time() - t_test
        total_test_time += t_test
        test_ap_list.append((test_ap,test_auc))
        emit_epoch_metrics(
            ctx=ctx,
            epoch=e,
            train_seconds=time_prep,
            batch_count=b_cnt,
            batch_latency_samples=batch_latency_samples,
            trainloader=trainloader,
            local_node=local_node,
            remote_node=remote_node,
            local_edge=local_edge,
            remote_edge=remote_edge,
            test_ap=test_ap,
            test_auc=test_auc,
            val_ap=ap,
            val_auc=auc,
        )
        early_stopper.early_stop_check(ap)
        early_stop = False
        trainloader.local_node = 0
        trainloader.remote_node = 0
        trainloader.local_edge = 0
        trainloader.remote_edge = 0
        mailbox.tot_comm_count = 0
        mailbox.tot_shared_count = 0
        value,counts = torch.unique(graph.edge_index.reshape(-1),return_counts = True)
        node_degree = torch.zeros(graph.num_nodes,dtype=torch.long)
        value = value.to('cpu')
        counts = counts.to('cpu')
        node_degree[value] = counts
        if dist.get_world_size()==1:
            mailbox.mon.draw(node_degree,args.dataname,args.model,e)
            mailbox.mon.set_zero()            
        #mailbox.mon.draw(node_degree,args.dataname,e)
        #mailbox.mon.set_zero()
        loss_list.append(total_loss)
        val_list.append(ap)

        if early_stop:
            dist.barrier()
            print("Early stopping at epoch {:d}\n".format(e))
            print(f"Loading the best model at epoch {early_stopper.best_epoch}\n")
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            model.module.load_state_dict(torch.load(best_model_path))
            break
        else:
            print('\ttrain loss:{:.4f}  train ap:{:4f}  val ap:{:4f}  val auc:{:4f} test ap {:4f} test auc{:4f}\n'.format(total_loss,train_ap, ap, auc,test_ap,test_auc))
            print('\ttotal time:{:.2f}s  prep time:{:.2f}s\n test time {:.2f}'.format(time.time()-epoch_start_time, time_prep,t_test))    
            print('alpha is {} beta is {}'.format(ada_param.alpha,ada_param.beta))
            torch.save(model.module.state_dict(), get_checkpoint_path(e))
        if args.model == 'TGN':
            pass
#            print('weight {} {}\n'.format(tt.weight_count_local,tt.weight_count_remote))
#            print('ssim {} {}\n'.format(tt.ssim_local/tt.ssim_cnt,tt.ssim_remote/tt.ssim_cnt))
    if not args.metrics_path:
        torch.save(val_list,'all_{}/{}/{}/val_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.seed,args.dataname,args.model,args.partition,args.topk,dist.get_world_size(),dist.get_rank(),args.sample_type,args.probability,args.memory_type,args.shared_memory_ssim))
        torch.save(loss_list,'all_{}/{}/{}/loss_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.seed,args.dataname,args.model,args.partition,args.topk,dist.get_world_size(),dist.get_rank(),args.sample_type,args.probability,args.memory_type,args.shared_memory_ssim))
        torch.save(test_ap_list,'all_{}/{}/{}/test_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.seed,args.dataname,args.model,args.partition,args.topk,dist.get_world_size(),dist.get_rank(),args.sample_type,args.probability,args.memory_type,args.shared_memory_ssim))
    
    print(avg_time)
    if not early_stop:        
        dist.barrier()
        print(f"Loading the best model at epoch {early_stopper.best_epoch}")
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        model.module.load_state_dict(torch.load(best_model_path)) 
    print('best test AP:{:4f} test auc{:4f}'.format(*test_ap_list[early_stopper.best_epoch]))
    val_list = torch.tensor(val_list)
    loss_list = torch.tensor(loss_list)
    print('test_dataset {} avg_time {} test time {}\n'.format(test_data.edges.shape[1],avg_time/epoch_cnt,total_test_time/epoch_cnt))
    torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
    ctx.shutdown()


if __name__ == "__main__":
    main()
