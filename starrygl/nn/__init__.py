from .convs import GCNConv, GATConv, GINConv
from .basic_gnn import BasicGNN

class GCN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)
    

class GAT(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return GATConv(in_channels, out_channels, **kwargs)
    
class GIN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return GINConv(in_channels, out_channels, **kwargs)