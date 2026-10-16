from ..imports import * 

__all__ = [
    'LinkPredictor', 
]


class LinkPredictor(nn.Module):
    def __init__(self, 
                 *, 
                 in_dim: int, 
                 hidden_dim: int,  
                 out_dim: int = 1, 
                 num_layers: int,
                 activation: nn.Module = nn.ReLU(), 
                 dropout: float = 0.):
        super().__init__()
        
        assert num_layers >= 2 
        self.num_layers = num_layers 
        assert out_dim == 1 

        self.fc_list = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim), 
            *[
                nn.Linear(hidden_dim, hidden_dim) 
                for _ in range(num_layers - 2) 
            ],
            nn.Linear(hidden_dim, out_dim), 
        ])
        
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 

    def forward(self, 
                src_feat: FloatTensor, 
                dest_feat: FloatTensor) -> FloatTensor: 
        num_edges, feat_dim = src_feat.shape 
        assert dest_feat.shape == (num_edges, feat_dim)
                
        h = src_feat * dest_feat 
        assert h.shape == (num_edges, feat_dim)
                
        for l in range(self.num_layers):
            h = self.fc_list[l](h) 
            
            if l < self.num_layers - 1: 
                h = self.activation(h) 
                h = self.dropout(h) 
                
        out = torch.sigmoid(h).squeeze(-1) 
        assert out.shape == (num_edges,) 
                
        return out 
