from .convs import GCNConv, GATConv, GINConv
from .convs import ShrinkGCNConv, ShrinkGATConv, ShrinkGINConv
from .basic_gnn import BasicGNN, BasicLayerOptions, BasicInputOptions, BasicJKOptions, BasicStraightOptions

class GCN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return ShrinkGCNConv(in_channels, out_channels, **kwargs)
    

class GAT(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return ShrinkGATConv(in_channels, out_channels, **kwargs)
    
class GIN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return ShrinkGINConv(in_channels, out_channels, **kwargs)
    