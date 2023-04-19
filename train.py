import torch
import torch.nn as nn

from torch import Tensor
from typing import *

import os
import time
import psutil

from starrygl.nn import *
from starrygl.graph import DistGraph

from starrygl.parallel import init_process_group, convert_parallel_model
from starrygl.parallel import compute_gcn_norm, SyncBatchNorm, with_nccl

from starrygl.utils import train_epoch, eval_epoch, partition_load, main_print



if __name__ == "__main__":
    # 启动分布式进程组，并分配计算设备
    device = init_process_group(backend="nccl")
    
    # 加载数据集
    pdata = partition_load("./cora", algo="metis").to(device)
    g = DistGraph(ids=pdata.ids, edge_index=pdata.edge_index)
    
    # g.args["async_op"] = True
    g.args["sample_k"] = 20
    
    g.edata["gcn_norm"] = compute_gcn_norm(g)
    g.ndata["x"] = pdata.x
    g.ndata["y"] = pdata.y
    
    # 定义GAT图神经网络模型
    net = GCN(
        g=g,
        layer_options=BasicLayerOptions(
            in_channels=pdata.num_features,
            hidden_channels=64,
            num_layers=2,
            out_channels=pdata.num_classes,
            norm="batchnorm",
        ),
        input_options=BasicInputOptions(
            straight_enabled=True,
        ),
        jk_options=BasicJKOptions(
            jk_mode=None,
        ),
        straight_options=BasicStraightOptions(
            enabled=True,
        ),
    ).to(device)
    
    # 转换成分布式并行版本
    net = convert_parallel_model(net)
    
    # 定义优化器
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    
    avg_mem = 0.0
    avg_dur = 0.0
    avg_num = 0

    # 开始训练
    best_val_acc = best_test_acc = 0
    for ep in range(1, 10+1):
        time_start = time.time()

        train_loss, train_acc = train_epoch(net, opt, g, pdata.train_mask)
        val_loss, val_acc = eval_epoch(net, g, pdata.val_mask)
        test_loss, test_acc = eval_epoch(net, g, pdata.test_mask)
        # val_loss, val_acc = train_loss, train_acc
        # test_loss, test_acc = train_loss, train_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        duration = time.time() - time_start
        if with_nccl():
            cur_mem = torch.cuda.memory_reserved()
        else:
            cur_mem = psutil.Process(os.getpid()).memory_info().rss
        cur_mem_mb = round(cur_mem / 1024**2)

        if ep > 1:
            avg_mem += cur_mem
            avg_dur += duration
            avg_num += 1
                
        main_print(
            f"ep: {ep}, mem: {cur_mem_mb}MiB, duration: {duration:.2f}s, "
            f"loss: [{train_loss:.4f}/{val_loss:.4f}/{test_loss:.6f}], "
            f"accuracy: [{train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}], "
            f"best_accuracy: {best_test_acc:.4f}")
    avg_mem = round(avg_mem / avg_num / 1024**2)
    avg_dur = avg_dur / avg_num
    main_print(f"average memory: {avg_mem}MiB, average duration: {avg_dur:.2f}s")
    