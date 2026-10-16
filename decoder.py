import torch 
import torch.nn as nn
from torch import Tensor

import graph_util


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sequence_len: int,
        share_projector: bool,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sequence_len = sequence_len        

        mlp = lambda: graph_util.MLP(
            in_dim = in_dim,
            out_dim = out_dim,
            num_layers = num_layers,
            dropout = dropout,
        )
        
        if share_projector:
            self.projector_list = nn.ModuleList([mlp()] * sequence_len)
        else:
            self.projector_list = nn.ModuleList([
                mlp()
                for _ in range(sequence_len)
            ])
        assert len(self.projector_list) == sequence_len 
        
    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        batch_size = len(input)
        assert input.shape == (batch_size, self.sequence_len, self.in_dim)
        
        output_list = []
        
        for i in range(self.sequence_len):
            output = self.projector_list[i]( input[:, i, :] )
            assert output.shape == (batch_size, self.out_dim)
            
            output_list.append(output)
            
        output = torch.stack(output_list, dim=1)
        assert output.shape == (batch_size, self.sequence_len, self.out_dim)
        
        return output 
