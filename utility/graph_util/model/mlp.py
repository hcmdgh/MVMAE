import torch 
import torch.nn as nn
from torch import Tensor

from .nn import Linear 


class MLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = -1,
        out_dim: int,
        num_layers: int,
        bias: bool = True,
        is_last_layer: bool = True,
        batch_norm: bool = False,
        activation: str = 'relu',
        dropout: float = 0.,
    ):
        super().__init__()

        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.num_layers = num_layers

        if hidden_dim <= 0:
            self.hidden_dim = hidden_dim = (in_dim + out_dim) // 2 
        else:
            self.hidden_dim = hidden_dim

        if num_layers > 1:
            linear_list = [
                Linear(
                    in_dim = in_dim,
                    out_dim = hidden_dim,
                    bias = bias,
                    batch_norm = batch_norm,
                    activation = activation,
                    dropout = dropout,
                ),
                *[
                    Linear(
                        in_dim = hidden_dim,
                        out_dim = hidden_dim,
                        bias = bias,
                        batch_norm = batch_norm,
                        activation = activation,
                        dropout = dropout,
                    )
                    for _ in range(num_layers - 2)
                ],
                Linear(
                    in_dim = hidden_dim,
                    out_dim = out_dim,
                    bias = bias,
                    batch_norm = False if is_last_layer else batch_norm,
                    activation = 'none' if is_last_layer else activation,
                    dropout = 0. if is_last_layer else dropout,
                ),
            ]
        elif num_layers == 1:
            linear_list = [
                Linear(
                    in_dim = in_dim,
                    out_dim = out_dim,
                    bias = bias,
                    batch_norm = False if is_last_layer else batch_norm,
                    activation = 'none' if is_last_layer else activation,
                    dropout = 0. if is_last_layer else dropout,
                )
            ]
        else:
            raise ValueError 

        assert len(linear_list) == self.num_layers 

        self.sequential = nn.Sequential(*linear_list)

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.sequential(input) 
