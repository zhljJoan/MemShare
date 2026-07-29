import argparse
import json
import math
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
from starrygl.sample.core_memshare import (
    CommunicationAwareNegativeSampling,
    LastFetchedMemoryPredictor,
    ProbabilityAwareEdgeCache,
    UnifiedRequestPlanner,
    importance_weighted_bce_with_logits,
)
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
parser.add_argument('--core_min_probability', default=0.05, type=float,
                    help='minimum residual-neighbor exploration probability')
parser.add_argument('--core_deterministic_recent', default=0, type=int,
                    help='always retain this many newest candidate neighbors')
parser.add_argument('--core_group_size', default=8, type=int,
                    help='roots per coordinated-sampling random-number group')
parser.add_argument('--core_uncertainty_weight', default=1.0, type=float)
parser.add_argument('--core_age_weight', default=0.25, type=float)
parser.add_argument('--core_edge_error_weight', default=0.5, type=float)
parser.add_argument('--core_age_scale', default=1.0, type=float)
parser.add_argument('--core_memory_cost', default=1.0, type=float)
parser.add_argument('--core_edge_cost', default=1.0, type=float)
parser.add_argument('--core_cached_edge_cost', default=0.0, type=float)
parser.add_argument('--core_request_cost', default=0.05, type=float)
parser.add_argument('--core_memory_cache_capacity', default=200000, type=int,
                    help='number of last-fetched node memories retained')
parser.add_argument('--core_edge_cache_ratio', default=0.0, type=float,
                    help='fraction of global edges admitted to the static cache')
parser.add_argument('--core_edge_cache_mb', default=0.0, type=float,
                    help='optional hard memory limit for the edge cache')
parser.add_argument('--core_cache_variance_weight', default=1.0, type=float)
parser.add_argument('--core_cache_min_residency', default=8, type=int)
parser.add_argument('--core_cache_max_replacements', default=64, type=int)
parser.add_argument('--core_neg_pool_factor', default=1, type=int,
                    help='candidate multiplier; values above one enable hard negatives')
parser.add_argument('--core_neg_hardness_beta', default=1.0, type=float)
parser.add_argument('--core_neg_exploration', default=0.1, type=float)
parser.add_argument('--core_neg_embedding_capacity', default=200000, type=int)
parser.add_argument('--core_neg_local_cost', default=0.05, type=float)
parser.add_argument('--core_neg_cached_cost', default=0.02, type=float)
parser.add_argument('--core_neg_reuse_cost', default=0.01, type=float)
parser.add_argument('--core_neg_remote_cost', default=1.0, type=float)
parser.add_argument('--core_request_planner', action='store_true',
                    help='collect typed, owner-bucketed request-plan statistics')
parser.add_argument('--core_smoke_batches', default=0, type=int,
                    help='run only this many training batches and skip evaluation')
parser.add_argument('--core_metrics_path', default='', type=str,
                    help='write rank-aggregated experiment metrics as JSON')
parser.add_argument('--core_experiment_name', default='', type=str,
                    help='stable label included in structured experiment metrics')
parser.add_argument(
    '--core_freeze_adaptive_parameters',
    action='store_true',
    help='hold sampling probability and smooth-sync threshold fixed for comparisons',
)
parser.add_argument(
    '--core_disable_memory_predictor',
    action='store_true',
    help='run the native communication-aware sampler with conservative static state',
)
parser.add_argument(
    '--core_screen_epochs',
    default=0,
    type=int,
    help='run this many train/validation/test epochs without writing checkpoints',
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


def _sum_metric(records, section, key):
    return sum(float(record.get(section, {}).get(key, 0.0)) for record in records)


def _aggregate_core_metrics(records, *, epoch, edge_feature_bytes):
    total_batches = sum(int(record["batches"]) for record in records)
    total_loss = sum(float(record["total_loss"]) for record in records)
    planner_keys = (
        "batches",
        "raw_node_requests",
        "raw_edge_requests",
        "unique_node_requests",
        "unique_edge_requests",
        "remote_node_requests",
        "remote_edge_requests",
        "cached_edge_requests",
    )
    planner = {
        key: _sum_metric(records, "request_planner", key)
        for key in planner_keys
    }
    planner["node_dedup_ratio"] = (
        1.0 - planner["unique_node_requests"] / planner["raw_node_requests"]
        if planner["raw_node_requests"]
        else 0.0
    )
    planner["edge_dedup_ratio"] = (
        1.0 - planner["unique_edge_requests"] / planner["raw_edge_requests"]
        if planner["raw_edge_requests"]
        else 0.0
    )
    planner["estimated_remote_edge_feature_bytes"] = (
        planner["remote_edge_requests"] * edge_feature_bytes
    )

    cache_keys = (
        "capacity",
        "size",
        "requests",
        "hits",
        "bytes_saved",
        "insertions",
        "evictions",
        "rejected",
        "memory_bytes",
    )
    cache = {
        key: _sum_metric(records, "edge_cache", key)
        for key in cache_keys
    }
    cache["object_hit_rate"] = (
        cache["hits"] / cache["requests"] if cache["requests"] else 0.0
    )

    predictor_keys = (
        "entries",
        "owners",
        "observations",
        "last_hit_count",
        "centroid_hit_count",
        "zero_fallback_count",
    )
    predictor = {
        key: _sum_metric(records, "memory_predictor", key)
        for key in predictor_keys
    }
    predictor["mean_observed_residual"] = (
        sum(
            float(record.get("memory_predictor", {}).get("observations", 0.0))
            * float(
                record.get("memory_predictor", {}).get(
                    "mean_observed_residual", 0.0
                )
            )
            for record in records
        )
        / predictor["observations"]
        if predictor["observations"]
        else 0.0
    )

    negative_sum_keys = (
        "total_samples",
        "total_remote_samples",
        "total_reused_samples",
        "total_hardness",
        "total_communication_cost",
        "total_importance_weight",
        "total_squared_importance_weight",
        "total_clipped_samples",
    )
    negative = {
        key: _sum_metric(records, "negative_sampler", key)
        for key in negative_sum_keys
    }
    negative["pool_factor"] = max(
        (
            float(record.get("negative_sampler", {}).get("pool_factor", 1.0))
            for record in records
        ),
        default=1.0,
    )
    negative["target_localism_probability"] = max(
        (
            float(
                record.get("negative_sampler", {}).get(
                    "target_localism_probability", 1.0
                )
            )
            for record in records
        ),
        default=1.0,
    )
    negative["mean_local_destination_fraction"] = (
        sum(
            float(
                record.get("negative_sampler", {}).get(
                    "local_destination_fraction", 0.0
                )
            )
            for record in records
        )
        / len(records)
        if records
        else 0.0
    )
    negative["embedding_entries"] = _sum_metric(
        records, "negative_sampler", "embedding_entries"
    )
    negative["mean_hardness"] = (
        negative["total_hardness"] / negative["total_samples"]
        if negative["total_samples"]
        else 0.0
    )
    negative["mean_communication_cost"] = (
        negative["total_communication_cost"] / negative["total_samples"]
        if negative["total_samples"]
        else 0.0
    )
    negative["mean_importance_weight"] = (
        negative["total_importance_weight"] / negative["total_samples"]
        if negative["total_samples"]
        else 0.0
    )
    negative["effective_sample_size"] = (
        negative["total_importance_weight"] ** 2
        / negative["total_squared_importance_weight"]
        if negative["total_squared_importance_weight"]
        else 0.0
    )
    negative["effective_sample_size_ratio"] = (
        negative["effective_sample_size"] / negative["total_samples"]
        if negative["total_samples"]
        else 0.0
    )
    negative["remote_sample_ratio"] = (
        negative["total_remote_samples"] / negative["total_samples"]
        if negative["total_samples"]
        else 0.0
    )
    negative["reuse_sample_ratio"] = (
        negative["total_reused_samples"] / negative["total_samples"]
        if negative["total_samples"]
        else 0.0
    )
    negative["clipped_fraction"] = (
        negative["total_clipped_samples"] / negative["total_samples"]
        if negative["total_samples"]
        else 0.0
    )

    probability_count = _sum_metric(records, "sample_probability", "count")
    sample_probability = {
        "count": probability_count,
        "mean": (
            _sum_metric(records, "sample_probability", "sum")
            / probability_count
            if probability_count
            else 0.0
        ),
        "min": min(
            (
                float(record["sample_probability"]["min"])
                for record in records
                if record.get("sample_probability", {}).get("count", 0.0)
            ),
            default=0.0,
        ),
        "max": max(
            (
                float(record["sample_probability"]["max"])
                for record in records
                if record.get("sample_probability", {}).get("count", 0.0)
            ),
            default=0.0,
        ),
    }

    access_keys = (
        "local_nodes",
        "remote_nodes",
        "local_edges",
        "remote_edges",
        "memory_sync",
        "shared_sync",
    )
    access = {
        key: _sum_metric(records, "access", key)
        for key in access_keys
    }
    access["remote_node_ratio"] = (
        access["remote_nodes"]
        / (access["local_nodes"] + access["remote_nodes"])
        if access["local_nodes"] + access["remote_nodes"]
        else 0.0
    )
    access["remote_edge_ratio"] = (
        access["remote_edges"]
        / (access["local_edges"] + access["remote_edges"])
        if access["local_edges"] + access["remote_edges"]
        else 0.0
    )

    critical_path_seconds = max(
        (float(record["train_seconds"]) for record in records), default=0.0
    )
    report = {
        "schema_version": 1,
        "experiment": args.core_experiment_name,
        "dataset": args.dataname,
        "epoch": int(epoch),
        "world_size": len(records),
        "configuration": {
            "model": args.model,
            "partition": args.partition,
            "topk": args.topk,
            "sample_type": args.sample_type,
            "probability": args.probability,
            "memory_type": args.memory_type,
            "shared_memory_ssim": args.shared_memory_ssim,
            "seed": args.seed,
            "smoke_batches": args.core_smoke_batches,
            "edge_cache_ratio": args.core_edge_cache_ratio,
            "edge_cache_mb": args.core_edge_cache_mb,
            "negative_pool_factor": args.core_neg_pool_factor,
            "deterministic_recent": args.core_deterministic_recent,
            "coordinated_group_size": args.core_group_size,
            "request_planner": bool(args.core_request_planner),
            "freeze_adaptive_parameters": bool(
                args.core_freeze_adaptive_parameters
            ),
            "memory_predictor": not bool(
                args.core_disable_memory_predictor
            ),
            "screen_epochs": int(args.core_screen_epochs),
        },
        "result": {
            "status": "completed",
            "batches_per_rank": [int(record["batches"]) for record in records],
            "total_rank_batches": total_batches,
            "critical_path_seconds": critical_path_seconds,
            "mean_seconds_per_batch": (
                critical_path_seconds
                / min(int(record["batches"]) for record in records)
                if records and min(int(record["batches"]) for record in records)
                else 0.0
            ),
            "mean_loss": total_loss / total_batches if total_batches else None,
            "loss_is_finite": all(
                bool(record["loss_is_finite"]) for record in records
            ),
            "positive_examples": sum(
                int(record["positive_examples"]) for record in records
            ),
            "negative_examples": sum(
                int(record["negative_examples"]) for record in records
            ),
            "peak_gpu_memory_bytes": max(
                (int(record["peak_gpu_memory_bytes"]) for record in records),
                default=0,
            ),
            "edge_feature_bytes_per_object": int(edge_feature_bytes),
            "final_sampling_probabilities": [
                float(record["adaptive_parameter"]["beta"])
                for record in records
            ],
            "final_smoothing_thresholds": [
                float(record["adaptive_parameter"]["alpha"])
                for record in records
            ],
        },
        "access": access,
        "sample_probability": sample_probability,
        "request_planner": planner,
        "edge_cache": cache,
        "memory_predictor": predictor,
        "negative_sampler": negative,
        "evaluation": (
            records[0].get("evaluation", {}) if records else {}
        ),
        "ranks": records,
        "notes": [
            "Pipeline counters may include one prefetched request plan beyond consumed batches.",
            "Short-run metrics screen systems behavior and numerical stability; they do not establish AP/AUC quality.",
        ],
    }
    return report


def _write_core_metrics(
    *,
    ctx,
    epoch,
    batches,
    total_loss,
    train_seconds,
    positive_examples,
    negative_examples,
    sample_probability_sum,
    sample_probability_count,
    sample_probability_min,
    sample_probability_max,
    trainloader,
    mailbox,
    edge_cache,
    request_planner,
    memory_predictor,
    train_neg_sampler,
    edge_feature_bytes,
    ada_param,
    evaluation=None,
):
    if not args.core_metrics_path:
        return
    local_record = {
        "rank": int(dist.get_rank()),
        "batches": int(batches),
        "total_loss": float(total_loss),
        "loss_is_finite": math.isfinite(float(total_loss)),
        "train_seconds": float(train_seconds),
        "positive_examples": int(positive_examples),
        "negative_examples": int(negative_examples),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "access": {
            "local_nodes": float(trainloader.local_node),
            "remote_nodes": float(trainloader.remote_node),
            "local_edges": float(trainloader.local_edge),
            "remote_edges": float(trainloader.remote_edge),
            "memory_sync": float(mailbox.tot_comm_count if mailbox else 0),
            "shared_sync": float(mailbox.tot_shared_count if mailbox else 0),
        },
        "sample_probability": {
            "sum": float(sample_probability_sum),
            "count": float(sample_probability_count),
            "min": (
                float(sample_probability_min)
                if sample_probability_count
                else 0.0
            ),
            "max": (
                float(sample_probability_max)
                if sample_probability_count
                else 0.0
            ),
        },
        "edge_cache": edge_cache.stats() if edge_cache is not None else {},
        "request_planner": (
            request_planner.stats() if request_planner is not None else {}
        ),
        "memory_predictor": (
            memory_predictor.stats() if memory_predictor is not None else {}
        ),
        "negative_sampler": (
            train_neg_sampler.diagnostics()
            if hasattr(train_neg_sampler, "diagnostics")
            else {}
        ),
        "adaptive_parameter": {
            "alpha": float(ada_param.alpha),
            "beta": float(ada_param.beta),
        },
        "evaluation": dict(evaluation or {}),
    }
    records = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(records, local_record, group=ctx.gloo_group)
    if dist.get_rank() != 0:
        return
    report = _aggregate_core_metrics(
        records,
        epoch=epoch,
        edge_feature_bytes=edge_feature_bytes,
    )
    metrics_path = Path(args.core_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    print("CORE_METRICS_JSON {}".format(json.dumps(report, sort_keys=True)))


def main():   
    #torch.backends.cudnn.benchmark = True
    #torch.autograd.set_detect_anomaly(True)
    print('LOCAL RANK {}, RANK{}'.format(os.environ["LOCAL_RANK"],os.environ["RANK"]))
    use_cuda = True
    sample_param, memory_param, gnn_param, train_param = parse_config('../config/{}.yml'.format(args.model))
    if args.core_screen_epochs < 0:
        raise ValueError('--core_screen_epochs cannot be negative')
    if args.core_screen_epochs > 0:
        train_param['epoch'] = args.core_screen_epochs
    memory_param['mode'] = args.memory_type
    ctx = DistributedContext.init(backend="nccl", use_gpu=True,memory_group_num=1,cache_use_rpc=True)
    torch.set_num_threads(10)
    device_id = torch.cuda.current_device()
    if ((args.dataname =='GDELT') & (dist.get_world_size() <=4 )):
        graph,full_sampler_graph,train_mask,val_mask,test_mask,full_train_mask,cache_route = load_from_speed(args.dataname,seed=123457,top=args.topk,sampler_graph_add_rev=True, feature_device=torch.device('cpu'),partition_path=args.partition)#torch.device('cpu'))
    else:
        graph,full_sampler_graph,train_mask,val_mask,test_mask,full_train_mask,cache_route = load_from_speed(args.dataname,seed=123457,top=args.topk,sampler_graph_add_rev=True, feature_device=torch.device('cuda:{}'.format(ctx.local_rank)),partition_path=args.partition)#torch.device('cpu'))
    if(args.dataname=='GDELT'):
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
    core_sampling_enabled = policy_train == 'boundery_recent_core'
    core_param = {
        'min_probability': args.core_min_probability,
        'uncertainty_weight': args.core_uncertainty_weight,
        'age_weight': args.core_age_weight,
        'edge_error_weight': args.core_edge_error_weight,
        'age_scale': args.core_age_scale,
        'memory_cost': args.core_memory_cost,
        'edge_cost': args.core_edge_cost,
        'cached_edge_cost': args.core_cached_edge_cost,
        'request_cost': args.core_request_cost,
        'deterministic_recent': args.core_deterministic_recent,
        'group_size': args.core_group_size,
        'seed': args.seed,
    }
    if memory_param['type'] != 'none':
        mailbox = SharedMailBox(graph.ids.shape[0], memory_param, dim_edge_feat = graph.efeat.shape[1] if graph.efeat is not None else 0,
        shared_nodes_index=graph.shared_nids_list[ctx.memory_group_rank],device = torch.device('cuda:{}'.format(local_rank)),
        start_historical=(args.memory_type=='historical'),
        ada_param = ada_param)
    else:
        mailbox = None

    memory_predictor = None
    if (
        core_sampling_enabled
        and mailbox is not None
        and not args.core_disable_memory_predictor
    ):
        memory_predictor = LastFetchedMemoryPredictor(
            memory_param['dim_out'],
            args.core_memory_cache_capacity,
        )

    edge_cache = None
    if graph.efeat is not None and (
        args.core_edge_cache_ratio > 0 or args.core_edge_cache_mb > 0
    ):
        feature_bytes = (
            math.prod(graph.efeat.shape[1:])
            * graph.efeat.accessor.data.element_size()
        )
        capacity = int(graph.num_edges * args.core_edge_cache_ratio)
        if args.core_edge_cache_mb > 0:
            byte_capacity = int(
                args.core_edge_cache_mb * 1024 * 1024 / max(1, feature_bytes)
            )
            capacity = (
                min(capacity, byte_capacity) if capacity > 0 else byte_capacity
            )
        edge_cache = ProbabilityAwareEdgeCache(
            capacity=max(0, capacity),
            feature_shape=graph.efeat.shape[1:],
            dtype=graph.efeat.dtype,
            device=torch.device('cuda:{}'.format(local_rank)),
            variance_weight=args.core_cache_variance_weight,
            min_residency=args.core_cache_min_residency,
            max_replacements_per_batch=args.core_cache_max_replacements,
        )

    request_planner = None
    if (
        args.core_request_planner
        or core_sampling_enabled
        or edge_cache is not None
        or args.core_neg_pool_factor > 1
    ):
        request_planner = UnifiedRequestPlanner(local_rank=dist.get_rank())

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
        core_param=core_param,
    )
    eval_policy = (
        'recent'
        if core_sampling_enabled or args.core_screen_epochs > 0
        else policy_train
    )
    eval_sampler = NeighborSampler(
        num_nodes=graph.num_nodes,
        num_layers=num_layers,
        fanout=fanout,
        graph_data=eval_sample_graph,
        workers=10,
        policy=eval_policy,
        graph_name="eval",
        local_part=dist.get_rank(),
        edge_part=DistIndex(graph.eids_mapper).part,
        node_part=DistIndex(graph.nids_mapper).part,
        ada_param=ada_param,
        no_neg=no_neg,
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
    
    if args.core_neg_pool_factor > 1:
        train_neg_sampler = CommunicationAwareNegativeSampling(
            'triplet',
            amount=args.neg_samples,
            dst_node_list=full_dst.unique(),
            id_mapper=graph.nids_mapper,
            local_part=dist.get_rank(),
            pool_factor=args.core_neg_pool_factor,
            hardness_beta=args.core_neg_hardness_beta,
            exploration=args.core_neg_exploration,
            embedding_capacity=args.core_neg_embedding_capacity,
            local_cost=args.core_neg_local_cost,
            cached_cost=args.core_neg_cached_cost,
            reuse_cost=args.core_neg_reuse_cost,
            remote_cost=args.core_neg_remote_cost,
            localism_probability=float(ada_param.beta),
            seed=args.seed,
        )
    elif args.local_neg_sample:
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
                                        edge_cache=edge_cache,
                                        edge_communication_cost=args.core_edge_cost,
                                        request_planner=request_planner,
                                        memory_predictor=memory_predictor,
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
    smoke_batches_completed = 0
    smoke_total_loss = 0.0
    smoke_train_seconds = 0.0
    smoke_positive_examples = 0
    smoke_negative_examples = 0
    smoke_sample_probability_sum = 0.0
    smoke_sample_probability_count = 0
    smoke_sample_probability_min = float("inf")
    smoke_sample_probability_max = 0.0
    for e in range(train_param['epoch']):
        ada_param.train()
        model.module.memory_updater.empty_cache()
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
        positive_examples = 0
        negative_examples = 0
        sample_probability_sum = 0.0
        sample_probability_count = 0
        sample_probability_min = float("inf")
        sample_probability_max = 0.0
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
            positive_examples += int(pred_pos.numel())
            negative_examples += int(pred_neg.numel())
            for layer in mfgs:
                for block in layer:
                    if 'sample_prob' not in block.edata:
                        continue
                    probability = block.edata['sample_prob'].detach().float()
                    if probability.numel() == 0:
                        continue
                    sample_probability_sum += float(probability.sum().item())
                    sample_probability_count += int(probability.numel())
                    sample_probability_min = min(
                        sample_probability_min, float(probability.min().item())
                    )
                    sample_probability_max = max(
                        sample_probability_max, float(probability.max().item())
                    )
            if hasattr(train_neg_sampler, 'update_embeddings'):
                root_count = mfgs[0][0].num_dst_nodes()
                train_neg_sampler.update_embeddings(
                    mfgs[0][0].srcdata['ID'][:root_count],
                    model.module.embedding[:root_count],
                )
            #print(time_count.elapsed_event(t2))
            loss = creterion(pred_pos, torch.ones_like(pred_pos)) 
            if 'neg_importance_weight' in metadata:
                loss += importance_weighted_bce_with_logits(
                    pred_neg,
                    torch.zeros_like(pred_neg),
                    metadata['neg_importance_weight'],
                )
            elif args.local_neg_sample is False:
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
                    and not args.core_freeze_adaptive_parameters
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
            if (
                args.core_smoke_batches > 0
                and smoke_batches_completed + b_cnt
                >= args.core_smoke_batches
            ):
                break
        
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
        if dist.get_rank() == 0:
            if edge_cache is not None:
                print('core edge cache {}'.format(edge_cache.stats()))
            if request_planner is not None:
                print('core request planner {}'.format(request_planner.stats()))
            if memory_predictor is not None:
                print('core memory predictor {}'.format(memory_predictor.stats()))
            if hasattr(train_neg_sampler, 'diagnostics'):
                print('core negative sampler {}'.format(
                    train_neg_sampler.diagnostics()
                ))
        edge_feature_bytes = (
            math.prod(graph.efeat.shape[1:])
            * graph.efeat.accessor.data.element_size()
            if graph.efeat is not None
            else 0
        )
        if args.core_smoke_batches > 0:
            smoke_batches_completed += b_cnt
            smoke_total_loss += total_loss
            smoke_train_seconds += time_prep
            smoke_positive_examples += positive_examples
            smoke_negative_examples += negative_examples
            smoke_sample_probability_sum += sample_probability_sum
            smoke_sample_probability_count += sample_probability_count
            if sample_probability_count:
                smoke_sample_probability_min = min(
                    smoke_sample_probability_min, sample_probability_min
                )
                smoke_sample_probability_max = max(
                    smoke_sample_probability_max, sample_probability_max
                )
            if smoke_batches_completed < args.core_smoke_batches:
                continue
            _write_core_metrics(
                ctx=ctx,
                epoch=e,
                batches=smoke_batches_completed,
                total_loss=smoke_total_loss,
                train_seconds=smoke_train_seconds,
                positive_examples=smoke_positive_examples,
                negative_examples=smoke_negative_examples,
                sample_probability_sum=smoke_sample_probability_sum,
                sample_probability_count=smoke_sample_probability_count,
                sample_probability_min=smoke_sample_probability_min,
                sample_probability_max=smoke_sample_probability_max,
                trainloader=trainloader,
                mailbox=mailbox,
                edge_cache=edge_cache,
                request_planner=request_planner,
                memory_predictor=memory_predictor,
                train_neg_sampler=train_neg_sampler,
                edge_feature_bytes=edge_feature_bytes,
                ada_param=ada_param,
            )
            dist.barrier()
            if dist.get_rank() == 0:
                print(
                    'Training smoke run completed after {} batches'.format(
                        smoke_batches_completed
                    )
                )
            ctx.shutdown()
            return
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
        _write_core_metrics(
            ctx=ctx,
            epoch=e,
            batches=b_cnt,
            total_loss=total_loss,
            train_seconds=time_prep,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            sample_probability_sum=sample_probability_sum,
            sample_probability_count=sample_probability_count,
            sample_probability_min=sample_probability_min,
            sample_probability_max=sample_probability_max,
            trainloader=trainloader,
            mailbox=mailbox,
            edge_cache=edge_cache,
            request_planner=request_planner,
            memory_predictor=memory_predictor,
            train_neg_sampler=train_neg_sampler,
            edge_feature_bytes=edge_feature_bytes,
            ada_param=ada_param,
            evaluation={
                'validation_ap': float(ap),
                'validation_auc': float(auc),
                'test_ap': float(test_ap),
                'test_auc': float(test_auc),
            },
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
            if args.core_screen_epochs == 0:
                torch.save(model.module.state_dict(), get_checkpoint_path(e))
        if args.model == 'TGN':
            pass
#            print('weight {} {}\n'.format(tt.weight_count_local,tt.weight_count_remote))
#            print('ssim {} {}\n'.format(tt.ssim_local/tt.ssim_cnt,tt.ssim_remote/tt.ssim_cnt))
    if args.core_screen_epochs > 0:
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                'Training quality screen completed after {} epochs'.format(
                    args.core_screen_epochs
                )
            )
        ctx.shutdown()
        return
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
