from ..common import * 


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 num_heads: int,
                 attn_dropout: float):
        super().__init__()

        self.num_heads = num_heads 
        self.emb_dim = emb_dim 
        assert emb_dim % num_heads == 0 
        self.head_dim = emb_dim // num_heads 
        self.scale = self.head_dim ** -0.5

        self.Q_fc = nn.Linear(emb_dim, emb_dim)
        self.K_fc = nn.Linear(emb_dim, emb_dim)
        self.V_fc = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.out_fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, 
                Q: FloatTensor, 
                K: FloatTensor, 
                V: FloatTensor, 
                attn_bias: Optional[FloatTensor] = None) -> FloatTensor:
        batch_size, seq_len = Q.shape[0], Q.shape[1] 
        assert Q.shape == K.shape == V.shape == (batch_size, seq_len, self.emb_dim)

        Q = self.Q_fc(Q).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.K_fc(K).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.V_fc(V).view(batch_size, seq_len, self.num_heads, self.head_dim) 

        _Q = Q.transpose(1, 2) 
        assert _Q.shape == (batch_size, self.num_heads, seq_len, self.head_dim)
        _K = K.transpose(1, 2).transpose(2, 3) 
        assert _K.shape == (batch_size, self.num_heads, self.head_dim, seq_len)
            
        attn = torch.matmul(_Q, _K) * self.scale 
        assert attn.shape == (batch_size, self.num_heads, seq_len, seq_len)

        if attn_bias is not None: 
            attn.add_(attn_bias)

        attn = torch.softmax(attn, dim=-1) 

        attn = self.attn_dropout(attn) 
        
        _V = V.transpose(1, 2) 
        assert _V.shape == (batch_size, self.num_heads, seq_len, self.head_dim)
        
        out = torch.matmul(attn, _V) 
        assert out.shape == (batch_size, self.num_heads, seq_len, self.head_dim) 
        
        out = out.transpose(1, 2)
        assert out.shape == (batch_size, seq_len, self.num_heads, self.head_dim) 

        out = out.reshape(batch_size, seq_len, self.emb_dim) 
        
        out = self.out_fc(out) 

        return out 


def feed_forward_network(in_dim: int,
                         hidden_dim: int,
                         out_dim: int) -> nn.Module: 
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), 
        nn.GELU(), 
        nn.Linear(hidden_dim, out_dim), 
    )
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 num_heads: int, 
                 feedforward_hidden_dim: int, 
                 dropout: float, 
                 attn_dropout: float): 
        super().__init__()

        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.attn = MultiHeadAttention(
            emb_dim = embedding_dim, 
            num_heads = num_heads, 
            attn_dropout = attn_dropout, 
        )

        self.dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embedding_dim) 
        
        self.ffn = feed_forward_network(embedding_dim, feedforward_hidden_dim, embedding_dim) 

        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, 
                x: FloatTensor, 
                attn_bias: Optional[FloatTensor] = None) -> FloatTensor:
        out = self.attn_norm(x)
        out = self.attn(Q=out, K=out, V=out, attn_bias=attn_bias) 
        out = self.dropout(out)
        
        x2 = x + out  

        out = self.ffn_norm(x2)
        out = self.ffn(out)
        out = self.ffn_dropout(out) 
        
        out = x2 + out

        return out 


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 num_heads: int,
                 num_layers: int,  
                 feedforward_hidden_dim: int, 
                 dropout: float, 
                 attn_dropout: float): 
        super().__init__()

        self.num_layers = num_layers

        self.encoder_layer_list = nn.ModuleList([
            TransformerEncoderLayer(
                embedding_dim = embedding_dim, 
                num_heads = num_heads, 
                feedforward_hidden_dim = feedforward_hidden_dim, 
                dropout = dropout, 
                attn_dropout = attn_dropout, 
            )
            for _ in range(num_layers) 
        ])
        
    def forward(self, 
                x: FloatTensor) -> FloatTensor:
        h = x 
        
        for encoder_layer in self.encoder_layer_list: 
            h = encoder_layer(h) 

        return h 
