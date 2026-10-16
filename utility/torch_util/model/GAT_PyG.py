from torch_geometric.nn import GATConv 

from ..common import * 


class GAT_PyG(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int, 
                 num_layers: int,
                 num_heads: int,
                 attn_dropout: float = 0., 
                 activation: Callable = nn.ReLU(), 
                 dropout: float = 0.):
        super().__init__()

        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        self.num_layers = num_layers 
        self.num_heads = num_heads
        
        self.conv_list = nn.ModuleList([
            GATConv(
                in_channels = in_dim, 
                out_channels = hidden_dim, 
                heads = num_heads, 
                concat = True, 
                dropout = attn_dropout, 
            ), 
            *[
                GATConv(
                    in_channels = hidden_dim * num_heads, 
                    out_channels = hidden_dim, 
                    heads = num_heads, 
                    concat = True, 
                    dropout = attn_dropout, 
                )
                for _ in range(num_layers - 2) 
            ],
            GATConv(
                in_channels = hidden_dim * num_heads, 
                out_channels = out_dim, 
                heads = 1, 
                concat = True,  
                dropout = attn_dropout, 
            ), 
        ])
        assert len(self.conv_list) == num_layers >= 2 
        
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self,
                node_feat: FloatTensor,
                edge_index: EdgeIndex) -> FloatTensor: 
        assert node_feat.shape == (num_nodes := len(node_feat), self.in_dim) 
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1])
        
        h = node_feat
                
        for l in range(self.num_layers): 
            h = self.conv_list[l](edge_index=edge_index, x=h) 
            
            if l < self.num_layers - 1: 
                assert h.shape == (num_nodes, self.hidden_dim * self.num_heads)
                h = self.activation(h) 
                h = self.dropout(h) 
            else:
                assert h.shape == (num_nodes, self.out_dim)
                
        return h 
