from ..imports import * 
from ..graph import generate_message, aggregate_message, edge_softmax 

__all__ = [
    'GATConv', 
    'GAT', 
]


class GATConv(nn.Module):
    def __init__(self,
                 in_dim: Union[int, tuple[int, int]],
                 out_dim: int, 
                 num_heads: int, 
                 concat: bool, 
                 attn_dropout: float = 0.):
        super().__init__()
        
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.num_heads = num_heads 
        self.concat = concat 
        
        if isinstance(in_dim, int): 
            self.src_in_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
            self.dest_in_fc = self.src_in_fc 
        elif isinstance(in_dim, tuple): 
            self.src_in_fc = nn.Linear(in_dim[0], num_heads * out_dim, bias=False)
            self.dest_in_fc = nn.Linear(in_dim[1], num_heads * out_dim, bias=False)
        else:
            raise TypeError
        
        self.src_attn = Parameter(torch.zeros(1, num_heads, out_dim)) 
        nn.init.xavier_uniform_(self.src_attn)  
        
        self.dest_attn = Parameter(torch.zeros(1, num_heads, out_dim)) 
        nn.init.xavier_uniform_(self.dest_attn)  

        self.attn_dropout = nn.Dropout(attn_dropout) 

        if concat: 
            self.bias = Parameter(torch.zeros(num_heads * out_dim)) 
        else:
            self.bias = Parameter(torch.zeros(out_dim)) 
            
    def forward(self,
                edge_index: EdgeIndex, 
                src_feat: FloatTensor,
                dest_feat: FloatTensor) -> FloatTensor:
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
        assert src_feat.shape == (num_src_nodes := len(src_feat), self.in_dim)
        assert dest_feat.shape == (num_dest_nodes := len(dest_feat), self.in_dim)

        src_h = self.src_in_fc(src_feat) 
        src_h = src_h.view(num_src_nodes, self.num_heads, self.out_dim) 
        
        dest_h = self.dest_in_fc(dest_feat) 
        dest_h = dest_h.view(num_dest_nodes, self.num_heads, self.out_dim) 

        src_alpha = (src_h * self.src_attn).sum(-1) 
        assert src_alpha.shape == (num_src_nodes, self.num_heads) 
        
        dest_alpha = (dest_h * self.dest_attn).sum(-1) 
        assert dest_alpha.shape == (num_dest_nodes, self.num_heads) 
        
        src_alpha = generate_message(
            node_feat = src_alpha, 
            edge_index = edge_index, 
            from_dest = False, 
        )
        assert src_alpha.shape == (num_edges, self.num_heads) 
        
        dest_alpha = generate_message(
            node_feat = dest_alpha, 
            edge_index = edge_index, 
            from_dest = True, 
        )
        assert dest_alpha.shape == (num_edges, self.num_heads) 

        alpha = src_alpha + dest_alpha 
        alpha = F.leaky_relu(alpha, negative_slope=0.2) 
        alpha = edge_softmax(
            edge_index = edge_index, 
            edge_feat = alpha,
            num_nodes = num_dest_nodes,
        )
        alpha = self.attn_dropout(alpha) 
        assert alpha.shape == (num_edges, self.num_heads) 

        alpha = alpha.view(num_edges, self.num_heads, 1) 

        msg = generate_message(
            node_feat = src_h, 
            edge_index = edge_index, 
        )
        assert msg.shape == (num_edges, self.num_heads, self.out_dim)

        msg.mul_(alpha) 
        
        out = aggregate_message(
            message = msg, 
            edge_index = edge_index, 
            num_nodes = num_dest_nodes, 
            reduce = 'sum', 
        )
        assert out.shape == (num_dest_nodes, self.num_heads, self.out_dim) 
        
        if self.concat:
            out = out.view(num_dest_nodes, self.num_heads * self.out_dim)
        else: 
            out = out.mean(dim=1) 
            assert out.shape == (num_dest_nodes, self.out_dim) 
            
        out.add_(self.bias) 
        
        return out 


class GAT(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int, 
                 num_layers: int,
                 num_heads: int,
                 attn_dropout: float = 0., 
                 activation: nn.Module = nn.ELU(), 
                 dropout: float = 0.):
        super().__init__()

        self.num_layers = num_layers 
        
        self.conv_list = nn.ModuleList([
            GATConv(
                in_dim = in_dim, 
                out_dim = hidden_dim, 
                num_heads = num_heads, 
                concat = True, 
                attn_dropout = attn_dropout, 
            ), 
            *[
                GATConv(
                    in_dim = hidden_dim * num_heads, 
                    out_dim = hidden_dim, 
                    num_heads = num_heads, 
                    concat = True, 
                    attn_dropout = attn_dropout, 
                )
                for _ in range(num_layers - 2) 
            ],
            GATConv(
                in_dim = hidden_dim * num_heads, 
                out_dim = out_dim, 
                num_heads = num_heads, 
                concat = False,  
                attn_dropout = attn_dropout, 
            ), 
        ])
        assert len(self.conv_list) == num_layers >= 2 
        
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self,
                edge_index: EdgeIndex, 
                node_feat: FloatTensor) -> FloatTensor: 
        h = node_feat
                
        for l in range(self.num_layers): 
            h = self.conv_list[l](edge_index=edge_index, src_feat=h, dest_feat=h) 
            
            if l < self.num_layers - 1: 
                h = self.activation(h) 
                h = self.dropout(h) 
                
        return h 
