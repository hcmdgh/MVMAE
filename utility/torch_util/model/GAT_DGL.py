import dgl 
from dgl.nn.pytorch import GATConv 

from ..imports import * 

__all__ = [
    'GAT_DGL', 
]


class GAT_DGL(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int, 
                 num_heads: int,
                 num_layers: int,
                 add_self_loop: bool = True, 
                 attn_dropout: float = 0., 
                 activation: nn.Module = nn.ELU(), 
                 dropout: float = 0.):
        super().__init__()

        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim 
        self.num_layers = num_layers 
        self.num_heads = num_heads 
        self.add_self_loop = add_self_loop 
        
        if num_layers == 1:
            self.conv_list = nn.ModuleList([
                GATConv(
                    in_feats = in_dim, 
                    out_feats = out_dim, 
                    num_heads = num_heads, 
                    attn_drop = attn_dropout, 
                ) 
            ])
        elif num_layers >= 2:
            self.conv_list = nn.ModuleList([
                GATConv(
                    in_feats = in_dim, 
                    out_feats = hidden_dim, 
                    num_heads = num_heads, 
                    attn_drop = attn_dropout, 
                ), 
                *[
                    GATConv(
                        in_feats = hidden_dim * num_heads, 
                        out_feats = hidden_dim, 
                        num_heads = num_heads, 
                        attn_drop = attn_dropout, 
                    )
                    for _ in range(num_layers - 2) 
                ],
                GATConv(
                    in_feats = hidden_dim * num_heads, 
                    out_feats = out_dim, 
                    num_heads = num_heads, 
                    attn_drop = attn_dropout, 
                ), 
            ])
        else:
            raise AssertionError 
        
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self,
                edge_index: EdgeIndex, 
                node_feat: FloatTensor) -> FloatTensor: 
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
        assert node_feat.shape == (num_nodes := len(node_feat), self.in_dim) 
        
        g = dgl.graph(tuple(edge_index), num_nodes=num_nodes) 

        if self.add_self_loop: 
            g = dgl.add_self_loop(dgl.remove_self_loop(g)) 
                
        h = node_feat
                
        for l in range(self.num_layers): 
            h = self.conv_list[l](graph=g, feat=h)  
            
            if l < self.num_layers - 1:
                assert h.shape == (num_nodes, self.num_heads, self.hidden_dim)
                 
                h = torch.flatten(h, start_dim=1) 
                h = self.activation(h) 
                h = self.dropout(h) 
            else:
                assert h.shape == (num_nodes, self.num_heads, self.out_dim)
                
                h = torch.flatten(h, start_dim=1) 
                assert h.shape == (num_nodes, self.num_heads * self.out_dim)
                
        return h 
