import torch 
import torch.nn as nn 
from torch import Tensor 

from .nn import Activation


class LinearDict(nn.Module):
    def __init__(
        self,
        in_dim_dict: dict[str, int],
        out_dim: int,
        activation: str = 'none',
    ):
        super().__init__()

        self.linear_dict = nn.ModuleDict({
            node_type: nn.Linear(in_dim, out_dim)
            for node_type, in_dim in in_dim_dict.items()
        })

        self.activation = Activation(activation)

    def forward(
        self, 
        input_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        output_dict = {
            node_type: self.linear_dict[node_type](input)
            for node_type, input in input_dict.items()
        }

        output_dict = {
            node_type: self.activation(output)
            for node_type, output in output_dict.items()
        }

        return output_dict
