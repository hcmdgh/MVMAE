from torch_geometric.nn import SAGEConv

from ..common import * 


class GraphSAGE_PyG(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 batch_norm: bool = False, 
                 activation: Callable = nn.ReLU(), 
                 dropout: float = 0.):
        super().__init__()

        self.num_layers = num_layers 
        
        self.conv_list = nn.ModuleList([
            SAGEConv(
                in_channels = in_dim, 
                out_channels = hidden_dim,
            ), 
            *[
                SAGEConv(
                    in_channels = hidden_dim, 
                    out_channels = hidden_dim,
                )
                for _ in range(num_layers - 2) 
            ],
            SAGEConv(
                in_channels = hidden_dim, 
                out_channels = out_dim,
            ), 
        ])
        assert len(self.conv_list) == num_layers >= 2 
        
        if batch_norm: 
            self.bn_list = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) 
                for _ in range(num_layers - 1) 
            ])
        else:
            self.bn_list = None 
            
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self,
                edge_index: EdgeIndex, 
                node_feat: FloatTensor) -> FloatTensor: 
        h = node_feat
        
        for l in range(self.num_layers): 
            h = self.conv_list[l](x=h, edge_index=edge_index)          
            
            if l < self.num_layers - 1: 
                if self.bn_list is not None: 
                    h = self.bn_list[l](h) 
                h = self.activation(h) 
                h = self.dropout(h)
                            
        return h 
