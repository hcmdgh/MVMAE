from torch_geometric.nn import GCNConv

from ..common import * 


class GCN_PyG(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 batch_norm: bool = False, 
                 activation: Callable = nn.ReLU(), 
                 dropout: float = 0.):
        super().__init__()

        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        self.num_layers = num_layers 
        
        if num_layers == 1:
            self.conv_list = nn.ModuleList([
                GCNConv(
                    in_channels = in_dim, 
                    out_channels = out_dim,
                )
            ]) 
        elif num_layers >= 2:
            self.conv_list = nn.ModuleList([
                GCNConv(
                    in_channels = in_dim, 
                    out_channels = hidden_dim,
                ), 
                *[
                    GCNConv(
                        in_channels = hidden_dim, 
                        out_channels = hidden_dim,
                    ) 
                    for _ in range(num_layers - 2) 
                ],
                GCNConv(
                    in_channels = hidden_dim, 
                    out_channels = out_dim,
                ), 
            ])
        else:
            raise AssertionError 
        
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
                node_feat: FloatTensor,
                edge_index: EdgeIndex) -> FloatTensor: 
        assert node_feat.shape == (num_nodes := len(node_feat), self.in_dim)
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1])
        
        h = node_feat
        
        for l in range(self.num_layers): 
            h = self.conv_list[l](x=h, edge_index=edge_index)          
            
            if l < self.num_layers - 1: 
                assert h.shape == (num_nodes, self.hidden_dim)
                
                if self.bn_list is not None: 
                    h = self.bn_list[l](h) 
                h = self.activation(h) 
                h = self.dropout(h)
            else:
                assert h.shape == (num_nodes, self.out_dim)
                            
        return h 
