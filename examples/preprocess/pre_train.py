import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn import SAGEConv
import numpy as np

class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.float().reshape((-1, 1))))
        return output

    
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class TSAGELayer(nn.Module):
    def __init__(self, in_dim = 0, edge_dim = 0, time_dim = 0, h_feats = 0):
        super(TSAGELayer, self).__init__()
        assert in_dim + time_dim != 0 and h_feats != 0
        self.time_dim = time_dim
        self.time_enc = TimeEncode(time_dim)
        self.sage = SAGEConv(in_dim,h_feats,"mean"),
                    
    
    def forward(self,b):
        time_f =  self.time_enc(b.edata['dt'])
        time_f = torch.cat((torch.zeros(b.num_dst_nodes(),self.time_dim,dtype = time_f.dtype,
                                        device = time_f.device),time_f),dim = 1)
        if 'f' in b.edata:
            edge_f = torch.cat((torch.zeros(b.num_dst_nodes(),b.edata['f'].shape[1],
                                            dtype = b.edata['f'].dtype,device = b.edata['f'].device),
                                            b.edata['f']),dim = 1)
            if 'h' in b.srcdata:
                b.srcdata['h'] = torch.cat((b.srcnode['h'],edge_f,time_f),dim = 1)
            else:
                b.srcdata['h'] = torch.cat((edge_f,time_f),dim = 1)
        else:
            if 'h' in b.srcdata:
                b.srcdata['h'] = torch.cat((b.srcnode['h'],time_f),dim = 1)
            else:
                b.srcdata['h'] = time_f
        return F.relu(self.sage(b,b.src_data['h']))
    
class TSAGEModel(nn.Module):
    def __init__(self, num_layer, node_dim, edge_dim, time_dim, h_dim):
        super(TSAGEModel, self).__init__()
        self.num_layer = num_layer
        layer = []
        for i in range(num_layer):
            if i != 0:
                layer.append(TSAGELayer(h_dim,edge_dim,time_dim,h_dim))
            else:
                layer.append(TSAGELayer(node_dim,edge_dim,time_dim,h_dim))
        self.layers = layer

    def forward(self,mfgs):
        for l in range(len(mfgs)):
            for h in range(len(mfgs[l])):
                if l < self.num_layer - 1:
                    mfgs[l+1][h].srcdata['h'] = self.layers[l](self,mfgs[l][h])
                else:
                    return self.layers[l](self,mfgs[l][h])
        
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]
        

# Thumbnail credits: Link Prediction with Neo4j, Mark Needham
# sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'