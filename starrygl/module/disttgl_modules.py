import torch
import dgl
import time
from starrygl.module.memorys import *
from starrygl.module.disttgl_layers import *

class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, num_node=None, no_learn_node=False, combined=False, edge_classification=False, edge_classes=0):
        super(GeneralModel, self).__init__()
        self.edge_classification = edge_classification
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'smart':
                self.memory_updater = SmartMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node, num_node, no_learn_node=no_learn_node)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
        else:
            raise NotImplementedError
        if not self.edge_classification:
            self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        else:
            self.edge_classifier = torch.nn.Linear(2 * gnn_param['dim_out'], edge_classes)
    
    def forward(self, mfg, metadata = None,neg_samples=1, mode = 'triplet'):
        # import pdb; pdb.set_trace()
        # torch.cuda.synchronize()
        # t_s=time.time()
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfg)
        # torch.cuda.synchronize()
        # t_mem = time.time() - t_s
        # t_s = time.time()
        rst = self.layers['l0h0'](mfg)
        self.embedding = rst.detach().clone()
        if not self.edge_classification:
            return self.edge_predictor(rst[metadata['src_pos_index']],rst[metadata['dst_pos_index']],rst[metadata['dst_neg_index']])
        else:
            rst = torch.cat([rst[metadata['src_pos_index']], rst[metadata['dst_pos_index']]], dim=1)
            return self.edge_classifier(rst), None


class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x