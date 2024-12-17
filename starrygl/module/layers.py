from os.path import abspath, join, dirname
import os
import sys
from os.path import abspath, join, dirname

from starrygl.distributed.utils import DistIndex
sys.path.insert(0, join(abspath(dirname(__file__))))
import torch
import dgl
import math
import numpy as np
from starrygl.sample.count_static import time_count as tt

class TimeEncode(torch.nn.Module):

    def __init__(self, dim, alpha = None, beta= None, parameter_requires_grad: bool = True):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))
        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False
            if alpha is None:
                alpha = math.sqrt(dim)
            if beta is None:
                beta = math.sqrt(dim)
            self.w.weight = torch.nn.Parameter((torch.from_numpy(1.0 / alpha ** np.linspace(0, dim/beta, dim, dtype=np.float32))).reshape(dim, -1)) 
        else:
            self.w.weight = torch.nn.Parameter((torch.from_numpy(1.0 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1)) 

    def forward(self, t):
        output = torch.cos(self.w(t.float().reshape((-1, 1))))
        return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h_pos_src, h_pos_dst, h_neg_src=None,h_neg_dst=None, 
                neg_samples=1,mode='triplet'):
        h_pos_src = self.src_fc(h_pos_src)
        h_pos_dst = self.dst_fc(h_pos_dst)
        h_neg_dst = self.dst_fc(h_neg_dst)
        if mode == 'triplet':
            h_pos_edge = torch.nn.functional.relu(h_pos_src + h_pos_dst)
            
            h_neg_edge = torch.nn.functional.relu(h_pos_src.tile(neg_samples, 1) + h_neg_dst)
        else:
            h_neg_src = self.src_fc(h_neg_src)
            h_pos_edge = torch.nn.functional.relu(h_pos_src + h_pos_dst)
            h_neg_edge = torch.nn.functional.relu(h_neg_src + h_neg_dst)
            
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)

class FeedForward(torch.nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, dim_out,expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = torch.nn.Linear(dims, dim_out)
        else:
            self.linear_0 = torch.nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = torch.nn.Linear(int(expansion_factor * dims), dim_out)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = torch.nn.functional.dropout(x, p=self.dropout)
        return x
    
class MixerBlock(torch.nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self,dim_input,dim_out,dropout=0,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4,
                 use_single_layer=False):
        super(MixerBlock, self).__init__()
        self.token_layernorm = torch.nn.LayerNorm(dim_out)
        self.token_forward = FeedForward(dim_input,dim_out,token_expansion_factor,dim_out,dropout,use_single_layer)
        self.channel_layernorm = torch.nn.LayerNorm(dim_out)
        self.channel_forward = FeedForward(dim_out,dim_out,channel_expansion_factor, dropout, use_single_layer)
    def token_mixer(self,x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    def channel_mixer(self,x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x
    def forward(self,x):
        x = x + self.token_mixer(x) + self.channel_mixer(x)
        return x
class MixerMLP(torch.nn.Module):
    def __init__(self, per_graph_size, time_channels,
                    input_channels, hidden_channels, out_channels,
                    num_layers=2, dropout=0.5,
                    token_expansion_factor=0.5, 
                    channel_expansion_factor=4, 
                    module_spec=None, use_single_layer=False,
                    num_neighbor = None
                   ):
            super().__init__()
            self.per_graph_size = per_graph_size

            self.num_layers = num_layers
        

            self.layernorm = torch.nn.LayerNorm(hidden_channels)
            self.mlp_head = torch.nn.Linear(hidden_channels, out_channels)

            # inner layers
            self.mixer_blocks = torch.nn.ModuleList()
            for ell in range(num_layers):
                if module_spec is None:
                    self.mixer_blocks.append(
                        MixerBlock(per_graph_size, hidden_channels, 
                                   token_expansion_factor, 
                                   channel_expansion_factor, 
                                   dropout, module_spec=None, 
                                   use_single_layer=use_single_layer)
                    )
                else:
                    self.mixer_blocks.append(
                        MixerBlock(per_graph_size, hidden_channels, 
                                   token_expansion_factor, 
                                   channel_expansion_factor, 
                                   dropout, module_spec=module_spec[ell], 
                                   use_single_layer=use_single_layer)
                    )



            # init
            self.reset_parameters()
    def forward(self,b):
        pass
    def block_padding(self,b):
        edges = b.adjacency_matrix()
        print(edges)
        #numbe
        #src,indices = b.srcdata['ID'].
        pass
    def forward(self, b):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        self.block_padding(b)
        #return x
    

class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        self.device = b.device
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=self.device)
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=self.device))
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=self.device)
            K = torch.zeros((b.num_edges(), self.dim_out), device=self.device)
            V = torch.zeros((b.num_edges(), self.dim_out), device=self.device)
            if self.dim_node_feat > 0:
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=self.device)
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(b.srcdata['h'][b.edges()[0]])
                V = self.w_v(b.srcdata['h'][b.edges()[0]])
            elif self.dim_time == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f']], dim=1))
                #K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                #V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(time_feat)
                V = self.w_v(time_feat)
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.edges()[0]], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.edges()[0]], time_feat], dim=1))
                #K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                #V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f'], time_feat], dim=1))
                #Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                #K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                #V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            #print('Q {} \n K {} \n V {} \n'.format(Q,K,V))
            #att_sum = torch.sum(Q*K,dim=2)
            #att_v_max = dgl.ops.copy_e_max(b,att_sum)
            #att_e_sub_max = torch.exp(dgl.ops.e_sub_v(b,att_sum,att_v_max))
            #att = dgl.ops.e_div_v(b,att_e_sub_max,torch.clamp_min(dgl.ops.copy_e_sum(b,att_e_sub_max),1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            #tt.weight_count_remote+=torch.sum(att[DistIndex(b.srcdata['ID']).part[b.edges()[0]]!=torch.distributed.get_rank()]**2)
            #tt.weight_count_local+=torch.sum(att[DistIndex(b.srcdata['ID']).part[b.edges()[0]]==torch.distributed.get_rank()]**2)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            #V_local = V.clone()
            #V_remote = V.clone()
            #V_local[DistIndex(b.srcdata['ID']).part[b.edges()[0]]!=torch.distributed.get_rank()] = 0
            #V_remote[DistIndex(b.srcdata['ID']).part[b.edges()[0]]==torch.distributed.get_rank()] = 0
            #b.edata['v0'] = V_local
            #b.edata['v1'] = V_remote
            #b.update_all(dgl.function.copy_e('v0', 'm0'), dgl.function.sum('m0', 'h0'))
            #b.update_all(dgl.function.copy_e('v1', 'm1'), dgl.function.sum('m1', 'h1'))
            #if 'weight' in b.edata and self.training is True:
            #    with torch.no_grad():
            #        weight = b.edata['weight'].reshape(-1,1)#(b.edata['weight']/torch.sum(b.edata['weight']).item()).reshape(-1,1)
                    #weight = 
                #print(weight.max())
            #    b.edata['v'] = V*weight
            #else:
            #    weight = b.edata['weight'].reshape(-1,1)
            b.edata['v'] = V
            #print(torch.sum(torch.sum(((V-V*weight)**2))))
            b.update_all(dgl.function.copy_e('v', 'm'), dgl.function.sum('m', 'h'))
            #tt.ssim_local+=torch.sum(torch.cosine_similarity(b.dstdata['h'],b.dstdata['h0']))
            #tt.ssim_remote+=torch.sum(torch.cosine_similarity(b.dstdata['h'],b.dstdata['h1']))
            #tt.ssim_cnt += b.num_dst_nodes()
            #print('dst {}'.format(b.dstdata['h']))
            #b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            #b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        else:
            rst = b.dstdata['h']
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst
            