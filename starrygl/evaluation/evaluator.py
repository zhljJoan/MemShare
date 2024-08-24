"""
Evaluator Module for Dynamic Link Prediction
"""

import numpy as np
from sklearn.metrics import *
import math
import torch
from tgb.utils.info import DATA_EVAL_METRIC_DICT


class Evaluator:
    def __init__(self, k_value: int = 10, valid_metric_list = None):
        r"""
        Parameters:
            name: name of the dataset
            k_value: the desired 'k' value for calculating metric@k
        """
        self.k_value = k_value  # for computing `hits@k`
        if valid_metric_list is None:
            self.valid_metric_list = ['hits@', 'mrr', 'ap', 'auc']
        else:
             self.valid_metric_list = valid_metric_list

    def _eval_hits_and_mrr(self, y_pred_pos, y_pred_neg):
        y_pred_neg = y_pred_neg.reshape(-1,y_pred_pos.shape[0]).T
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        #print(optimistic_rank,pessimistic_rank)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hitsK_list = (ranking_list <= self.k_value).to(torch.float)
        mrr_list = 1./ranking_list.to(torch.float)
        return hitsK_list.mean(),mrr_list.mean()

    @staticmethod
    def _eval_ap_and_auc(y_pred_pos, y_pred_neg):
        y_pred = torch.cat((y_pred_pos.view(-1),y_pred_neg.view(-1))).detach().cpu().numpy()
        y_true = torch.cat((torch.ones(y_pred_pos.numel()),torch.zeros(y_pred_neg.numel()))).detach().cpu().numpy()
        ap= average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        return ap,auc
        
    def eval(self, 
             y_pred_pos,
             y_pred_neg) -> dict:
        r"""
        evaluate the link prediction task
        this method is callable through an instance of this object to compute the metric

        Parameters:
            input_dict: a dictionary containing "y_pred_pos", "y_pred_neg", and "eval_metric"
                        the performance metric is calculated for the provided scores
            verbose: whether to print out the computed metric
        
        Returns:
            perf_dict: a dictionary containing the computed performance metric
        """
        
        hitk,mrr = self._eval_hits_and_mrr(y_pred_pos, y_pred_neg)
        ap,auc = self._eval_ap_and_auc(y_pred_pos,y_pred_neg)
        return {f'hits@{self.k_value}': hitk.mean(),
                    'mrr': mrr.mean(),
                    'ap': ap,
                    'auc': auc}
    
