import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.distributed.rpc as rpc

from torch import Tensor
from typing import *

from torch_sparse import SparseTensor

from starrygl.loader import NodeLoader, NodeHandle, TensorBuffer, RouteContext
from starrygl.parallel import init_process_group, convert_parallel_model
from starrygl.utils import partition_load, main_print, sync_print

class SimpleGNNConv(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        x = self.linear(x)
        x = adj_t @ x
        return self.norm(x)

class SimpleGNN(nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv1 = SimpleGNNConv(in_channels, hidden_channels)
        self.conv2 = SimpleGNNConv(hidden_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, handle: NodeHandle, buffers: List[TensorBuffer]) -> Tensor:
        futs = [
            handle.get_src_feats(buffers[0]),
            handle.get_ext_feats(buffers[1]),
        ]

        with RouteContext() as ctx:
            x = futs[0].wait()
            x = self.conv1(x, handle.adj_t)

            x, f = handle.push_and_pull(x, futs[1], buffers[1])
            x = self.conv2(x, handle.adj_t)
            ctx.add_futures(f) # 等当前batch推理完成后需要等待所有futures完成

            x = self.fc_out(x)
        return x

if __name__ == "__main__":
    # 启动分布式进程组，并分配计算设备
    device = init_process_group(backend="nccl")
    
    # 加载数据集
    pdata = partition_load("./cora", algo="metis")
    loader = NodeLoader(pdata.ids, pdata.edge_index, device)

    # 创建历史缓存
    hidden_size = 64
    buffers: List[TensorBuffer] = [
        TensorBuffer(pdata.num_features, loader.src_size, loader.route),
        TensorBuffer(hidden_size, loader.src_size, loader.route),
    ]

    # 设置节点初始特征，并预同步到其它分区
    buffers[0].data[:loader.dst_size] = pdata.x
    buffers[0].broadcast()
    
    # 创建模型
    net = SimpleGNN(pdata.num_features, hidden_size, pdata.num_classes).to(device)
    net = convert_parallel_model(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 训练阶段
    for ep in range(1, 100+1):
        epoch_loss = 0.0
        net.train()
        for handle in loader.iter(128):
            fut_m = handle.get_dst_feats(pdata.train_mask)
            fut_y = handle.get_dst_feats(pdata.y)

            h = net(handle, buffers)

            train_mask = fut_m.wait()
            logits = h[train_mask]
            if logits.size(0) > 0:
                y = fut_y.wait()[train_mask]
                loss = nn.CrossEntropyLoss()(logits, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
        main_print(ep, epoch_loss)
    rpc.shutdown()



# import torch
# import torch.nn as nn

# from torch import Tensor
# from typing import *

# import os
# import time
# import psutil

# from starrygl.nn import *
# from starrygl.graph import DistGraph

# from starrygl.parallel import init_process_group, convert_parallel_model
# from starrygl.parallel import compute_gcn_norm, SyncBatchNorm, with_nccl

# from starrygl.utils import train_epoch, eval_epoch, partition_load, main_print, sync_print



# if __name__ == "__main__":
#     # 启动分布式进程组，并分配计算设备
#     device = init_process_group(backend="nccl")
    
#     # 加载数据集
#     pdata = partition_load("./cora", algo="metis").to(device)
#     g = DistGraph(ids=pdata.ids, edge_index=pdata.edge_index)
    
#     # g.args["async_op"] = True
#     # g.args["num_samples"] = 20
    
#     g.edata["gcn_norm"] = compute_gcn_norm(g)
#     g.ndata["x"] = pdata.x
#     g.ndata["y"] = pdata.y
    
#     # 定义GAT图神经网络模型
#     net = ShrinkGCN(
#         g=g,
#         layer_options=BasicLayerOptions(
#             in_channels=pdata.num_features,
#             hidden_channels=64,
#             num_layers=3,
#             out_channels=pdata.num_classes,
#             norm="batchnorm",
#         ),
#         input_options=BasicInputOptions(
#             straight_enabled=True,
#             straight_num_samples = 200,
#         ),
#         straight_options=BasicStraightOptions(
#             enabled=True,
#             num_samples = 20,
#             # beta=1.1,
#         ),
#     ).to(device)
    
#     # 转换成分布式并行版本
#     net = convert_parallel_model(net)
    
#     # 定义优化器
#     opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    
#     avg_mem = 0.0
#     avg_dur = 0.0
#     avg_num = 0

#     # 开始训练
#     best_val_acc = best_test_acc = 0
#     for ep in range(1, 50+1):
#         time_start = time.time()

#         train_loss, train_acc = train_epoch(net, opt, g, pdata.train_mask)
#         val_loss, val_acc = eval_epoch(net, g, pdata.val_mask)
#         test_loss, test_acc = eval_epoch(net, g, pdata.test_mask)
#         # val_loss, val_acc = train_loss, train_acc
#         # test_loss, test_acc = train_loss, train_acc

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_test_acc = test_acc
        
#         duration = time.time() - time_start
#         if with_nccl():
#             cur_mem = torch.cuda.memory_reserved()
#         else:
#             cur_mem = psutil.Process(os.getpid()).memory_info().rss
#         cur_mem_mb = round(cur_mem / 1024**2)

#         if ep > 1:
#             avg_mem += cur_mem
#             avg_dur += duration
#             avg_num += 1
                
#         main_print(
#             f"ep: {ep}, mem: {cur_mem_mb}MiB, duration: {duration:.2f}s, "
#             f"loss: [{train_loss:.4f}/{val_loss:.4f}/{test_loss:.6f}], "
#             f"accuracy: [{train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}], "
#             f"best_accuracy: {best_test_acc:.4f}")
#     avg_mem = round(avg_mem / avg_num / 1024**2)
#     avg_dur = avg_dur / avg_num
#     main_print(f"average memory: {avg_mem}MiB, average duration: {avg_dur:.2f}s")
    