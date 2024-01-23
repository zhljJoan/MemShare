
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from starrygl.evaluation.get_evalute_data import get_link_prediction_data

from starrygl.sample.sample_core.base import NegativeSampling
from starrygl.sample.part_utils.partition_tgnn import partition_save
parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--data_name', default='WIKI', type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--num_neg_sample', default=1, type=int, metavar='W',
                    help='number of negative samples')
parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
args = parser.parse_args()

data_name = args.data_name

full_graph = get_link_prediction_data(data_name, args.val_ratio, args.test_ratio)

edge_index_dict = {}
edge_index_dict['edata'] = full_graph.edge_index
#edge_index_dict['sample_data'] = full_graph.sample_graph['edge_index']

full_graph.edge_index_dict = edge_index_dict
edge_weight_dict = {}
edge_weight_dict['edata'] = 1
partition_save('/mnt/data/part_data/evaluate/'+data_name, full_graph, 1, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
partition_save('/mnt/data/part_data/evaluate/'+data_name, full_graph, 2, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
partition_save('/mnt/data/part_data/evaluate/'+data_name, full_graph, 4, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)
partition_save('/mnt/data/part_data/evalute/'+data_name, full_graph, 8, 'metis_for_tgnn',
               edge_weight_dict=edge_weight_dict)

