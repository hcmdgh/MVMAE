from ..imports import * 

__all__ = [
    'SIGN', 
]


class FeedForwardNet(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int, 
                 activation: nn.Module = nn.PReLU(), 
                 dropout: float = 0.,):
        super().__init__()
        
        self.num_layers = num_layers 
        
        self.fc_list = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim), 
            *[
                nn.Linear(hidden_dim, hidden_dim) 
                for _ in range(num_layers - 2) 
            ],
            nn.Linear(hidden_dim, out_dim), 
        ])
        assert len(self.fc_list) == num_layers >= 2 
        
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 

    def forward(self, 
                x: FloatTensor) -> FloatTensor:
        h = x 
        
        for l in range(self.num_layers): 
            h = self.fc_list[l](h) 
            
            if l < self.num_layers - 1: 
                h = self.activation(h) 
                h = self.dropout(h) 
        
        return h


class SIGN(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 seq_len: int, 
                 num_layers: int, 
                 activation: nn.Module = nn.PReLU(), 
                 dropout: float = 0.):
        super().__init__()
        
        self.in_dim = in_dim 
        self.seq_len = seq_len 
        
        self.ffn = FeedForwardNet(
            in_dim = in_dim * seq_len, 
            hidden_dim = (in_dim + out_dim) // 2 * seq_len, 
            out_dim = out_dim, 
            num_layers = num_layers, 
            activation = activation, 
            dropout = dropout, 
        )

    def forward(self, 
                node_feat_seq: FloatTensor) -> FloatTensor: 
        assert node_feat_seq.shape == (num_nodes := len(node_feat_seq), self.seq_len, self.in_dim) 
        
        node_feat_seq = node_feat_seq.view(num_nodes, self.seq_len * self.in_dim) 
        
        out = self.ffn(node_feat_seq) 
                
        return out 
