import torch
import torch.distributed as dist

from torch import Tensor, LongTensor
from typing import *

def _local_TP_FP_FN(pred: LongTensor, targ: LongTensor, num_classes: int) -> Tensor:
    TP, FP, FN = 0, 1, 2
    tmp = torch.empty(3, num_classes, dtype=torch.float32, device=pred.device)
    for c in range(num_classes):
        pred_c = (pred == c)
        targ_c = (targ == c)
        tmp[TP, c] = torch.count_nonzero(pred_c and targ_c)
        tmp[FP, c] = torch.count_nonzero(pred_c and not targ_c)
        tmp[FN, c] = torch.count_nonzero(not pred_c and targ_c)
    return tmp

def micro_f1(pred: LongTensor, targ: LongTensor, num_classes: int) -> float:
    tmp = _local_TP_FP_FN(pred, targ, num_classes).sum(dim=-1)
    dist.all_reduce(tmp)
    TP, FP, FN = tmp.tolist()
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return 2 * precision * recall / (precision + recall)

def macro_f1(pred: LongTensor, targ: LongTensor, num_classes: int) -> float:
    tmp = _local_TP_FP_FN(pred, targ, num_classes)
    dist.all_reduce(tmp)
    TP, FP, FN = tmp
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1.mean().item()

def accuracy(pred: LongTensor, targ: LongTensor) -> float:
    tmp = torch.empty(2, dtype=torch.float32, device=pred.device)
    tmp[0] = pred.eq(targ).count_nonzero()
    tmp[1] = pred.size(0)
    dist.all_reduce(tmp)
    
    a, b = tmp.tolist()
    return a / b

def all_reduce_loss(loss: Tensor, batch_size: int) -> float:
    tmp = torch.tensor([
        loss.item() * batch_size,
        batch_size
    ], dtype=torch.float32, device=loss.device)
    dist.all_reduce(tmp)
    cum_loss, n = tmp.tolist()
    return cum_loss / n
