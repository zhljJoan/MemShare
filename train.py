import torch
import torch.nn as nn

from torch import Tensor
from typing import *

from starrygl.graph.distgraph import DistGraph
from starrygl.nn import *
from starrygl.nn.parallel import init_process_group, convert_parallel_model

from starrygl.utils.printer import main_print
from starrygl.utils.partition import partition_load
from starrygl.utils.metrics import accuracy


if __name__ == "__main__":
    # 启动分布式进程组，并分配计算设备
    device = init_process_group(backend="nccl")
    
    # 加载数据集
    pdata = partition_load("./cora", algo="metis").to(device)
    g = DistGraph(ids=pdata.ids, edge_index=pdata.edge_index)
    
    # 定义GAT图神经网络模型
    net = GCN(
        in_channels=pdata.num_features,
        hidden_channels=512,
        num_layers=3,
        out_channels=pdata.num_classes,
        # norm="batchnorm",
        dropout=0.5,
        # heads=8,
    ).to(device)
    
    # 转换成分布式并行版本
    net = convert_parallel_model(net)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 开始训练
    for ep in range(100):
        ## 训练步
        net.train()
        with g.scoped_manager():
            g.ndata["x"] = pdata.x
            g.edata["gcn_norm"] = pdata.gcn_norm
            h = net(g)
           
        loss: Tensor = criterion(
            h[pdata.train_mask],
            pdata.y[pdata.train_mask],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## 验证步
        net.eval()
        with g.scoped_manager():
            g.ndata["x"] = pdata.x
            g.edata["gcn_norm"] = pdata.gcn_norm
            with torch.no_grad():
                pred = net(g).argmax(dim=-1)
                acc = accuracy(
                    pred[pdata.val_mask],
                    pdata.y[pdata.val_mask],
                )
                
        main_print(f"{ep}: {loss.item()} acc: {acc}")