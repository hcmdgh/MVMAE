from ..common import * 


class TransformerEncoderLayer_Torch(nn.Module):
    def __init__(self,
                 embedding_dim: int, 
                 feedforward_dim: int,
                 num_heads: int, 
                 dropout: float): 
        super().__init__()
        
        self.layer = nn.TransformerEncoderLayer(
            d_model = embedding_dim, 
            nhead = num_heads, 
            dim_feedforward = feedforward_dim, 
            dropout = dropout, 
            batch_first = True, 
        )
        
    def forward(self,
                src: FloatTensor) -> FloatTensor: 
        batch_size, seq_len, in_dim = src.shape 
        
        out = self.layer.forward(src=src) 
        assert out.shape == (batch_size, seq_len, in_dim) 
        
        return out 


class TransformerEncoder_Torch(nn.Module):
    def __init__(self,
                 embedding_dim: int, 
                 feedforward_dim: int,
                 num_heads: int, 
                 num_layers: int, 
                 dropout: float): 
        super().__init__() 
        
        self.embedding_dim = embedding_dim 
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer = TransformerEncoderLayer_Torch(
                embedding_dim = embedding_dim, 
                feedforward_dim = feedforward_dim, 
                num_heads = num_heads, 
                dropout = dropout, 
            ).layer,
            num_layers = num_layers, 
        )
        
    def forward(self,
                src: FloatTensor,
                src_key_padding_mask: Optional[BoolTensor] = None) -> FloatTensor: 
        batch_size, seq_len = src.shape[0], src.shape[1] 
        assert src.shape == (batch_size, seq_len, self.embedding_dim) 
        
        out = self.encoder.forward(src=src, src_key_padding_mask=src_key_padding_mask) 
        assert out.shape == (batch_size, seq_len, self.embedding_dim)  
        
        return out 
