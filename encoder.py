import torch 
import torch.nn as nn 
from torch import Tensor 

import graph_util 

from .transformer import TransformerEncoder 


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_type: str,
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
        self.encoder_type = encoder_type

        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                sequence_len = sequence_len,
                embedding_dim = embedding_dim,
                num_layers = num_layers,
                num_heads = num_heads,
                feedforward_dim = feedforward_dim,
                dropout = dropout,
                math_backend = math_backend,
            )
        elif encoder_type == 'mlp':
            self.encoder = graph_util.MLP(
                in_dim = sequence_len * embedding_dim,
                out_dim = sequence_len * embedding_dim,
                num_layers = num_layers,
                dropout = dropout,
            )
        else:
            raise ValueError 
        
    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        batch_size = len(input)
        assert input.shape == (batch_size, self.sequence_len, self.embedding_dim)

        if self.encoder_type == 'transformer':
            output = self.encoder.forward(input)
            assert output.shape == (batch_size, self.sequence_len, self.embedding_dim)
        elif self.encoder_type == 'mlp':
            input = input.reshape(batch_size, self.sequence_len * self.embedding_dim)
            output = self.encoder.forward(input)
            assert output.shape == (batch_size, self.sequence_len * self.embedding_dim)
            output = output.reshape(batch_size, self.sequence_len, self.embedding_dim)
        else:
            raise ValueError 
        
        return output 
