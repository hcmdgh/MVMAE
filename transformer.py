import torch 
import torch.nn as nn 
from torch import Tensor 
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        math_backend: bool,
    ):
        super().__init__()
        
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.math_backend = math_backend
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = embedding_dim,
                nhead = num_heads,
                dim_feedforward = feedforward_dim,
                dropout = dropout,
                batch_first = True,
            ),
            num_layers = num_layers,
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        batch_size = len(input)
        assert input.shape == (batch_size, self.sequence_len, self.embedding_dim)

        if self.math_backend:
            context = sdpa_kernel(backends=[SDPBackend.MATH])
        else:
            context = nullcontext()
        
        with context:
            output = self.encoder(input)
            assert output.shape == (batch_size, self.sequence_len, self.embedding_dim)
        
        return output 
