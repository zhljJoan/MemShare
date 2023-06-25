# import torch
# import torch.nn as nn

# from torch import Tensor
# from ..graph import DistGraph

# from typing import *
# from .metrics import *

# def train_epoch(
#     model: nn.Module,
#     opt: torch.optim.Optimizer,
#     g: DistGraph,
#     mask: Optional[Tensor] = None,
# ) -> float:
#     model.train()
#     criterion = nn.CrossEntropyLoss()

#     pred: Tensor = model(g)
#     targ: Tensor = g.ndata["y"]
    
#     if mask is not None:
#         pred = pred[mask]
#         targ = targ[mask]
    
#     loss: Tensor = criterion(pred, targ)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     with torch.no_grad():
#         train_loss = all_reduce_loss(loss, targ.size(0))
#         train_acc  = accuracy(pred.argmax(dim=-1), targ)
#     return train_loss, train_acc

# @torch.no_grad()
# def eval_epoch(
#     model: nn.Module,
#     g: DistGraph,
#     mask: Optional[Tensor] = None,
# ) -> Tuple[float, float]:
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
    
#     pred: Tensor = model(g)
#     targ: Tensor = g.ndata["y"]
    
#     if mask is not None:
#         pred = pred[mask]
#         targ = targ[mask]
    
#     loss = criterion(pred, targ)
    
#     eval_loss = all_reduce_loss(loss, targ.size(0))
#     eval_acc = accuracy(pred.argmax(dim=-1), targ)
#     return eval_loss, eval_acc