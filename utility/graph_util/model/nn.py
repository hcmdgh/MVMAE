import torch 
import torch.nn as nn
from torch import Tensor


class BatchNorm(nn.Module):
    def __init__(
        self,
        in_dim: int,
        enable: bool = True,
    ):
        super().__init__()
        
        self.in_dim = in_dim 
        self.enable = enable 
        
        if enable:
            self.bn = nn.BatchNorm1d(in_dim)
        else:
            self.bn = nn.Identity()
            
    def forward(
        self, 
        input: Tensor,
    ) -> Tensor:
        input_shape = input.shape 
        assert input.ndim >= 2 and input.shape[-1] == self.in_dim 

        input = input.reshape(-1, self.in_dim)
        
        output = self.bn(input)
        assert output.shape == input.shape

        output = output.reshape(input_shape)
        
        return output 
    

class Activation(nn.Module):
    def __init__(self, mode: str):
        super().__init__()

        if mode == 'relu':
            self.activation = nn.ReLU()
        elif mode == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif mode == 'tanh':
            self.activation = nn.Tanh()
        elif mode == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif mode == 'elu':
            self.activation = nn.ELU()
        elif mode == 'gelu':
            self.activation = nn.GELU()
        elif mode == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError

    def forward(
        self, 
        input: Tensor,
    ) -> Tensor:
        return self.activation(input)


class Residual(nn.Module):
    def __init__(
        self,
        in_dim: int,
        residual_in_dim: int,
        mode: str,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.residual_in_dim = residual_in_dim
        self.mode = mode
        
        if mode == 'none':
            self.linear = None
        elif mode == 'add':
            assert residual_in_dim == in_dim
            self.linear = None
        elif mode == 'lin_add':
            self.linear = nn.Linear(residual_in_dim, in_dim)
        elif mode == 'concat_lin':
            self.linear = nn.Linear(in_dim + residual_in_dim, in_dim)
        elif mode == 'concat_lin_add':
            assert residual_in_dim == in_dim
            self.linear = nn.Linear(in_dim * 2, in_dim)
        else:
            raise ValueError
        
    def forward(
        self,
        input: Tensor,
        residual_input: Tensor,
    ) -> Tensor:
        batch_size = len(input) 
        assert input.shape == (batch_size, self.in_dim) 
        assert residual_input.shape == (batch_size, self.residual_in_dim)  
        
        if self.mode == 'none':
            return input
        elif self.mode == 'add':
            assert self.residual_in_dim == self.in_dim
            return input + residual_input
        elif self.mode == 'lin_add':
            assert self.linear
            return input + self.linear(residual_input)
        elif self.mode == 'concat_lin':
            assert self.linear
            return self.linear(torch.cat([input, residual_input], dim=-1))
        elif self.mode == 'concat_lin_add':
            assert self.linear
            assert self.residual_in_dim == self.in_dim
            return self.linear(torch.cat([input, residual_input], dim=-1)) + residual_input 
        else:
            raise ValueError


class NonLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        batch_norm: bool,
        activation: str,
        dropout: float,
    ):
        super().__init__()

        self.in_dim = in_dim

        self.batch_norm = BatchNorm(in_dim=in_dim, enable=batch_norm)

        self.activation = Activation(activation)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        output = self.batch_norm(input)
        output = self.activation(output)
        output = self.dropout(output)

        return output


class NonLinearWithResidual(nn.Module):
    def __init__(
        self,
        in_dim: int,
        residual_in_dim: int,
        batch_norm: bool = False,
        activation: str = 'relu',
        dropout: float = 0.,
        residual_type: str = 'none',
        residual_position: str = 'before',
    ):
        super().__init__()

        self.in_dim = in_dim
        self.residual_position = residual_position

        self.batch_norm = BatchNorm(in_dim=in_dim, enable=batch_norm)

        self.activation = Activation(activation)

        self.dropout = nn.Dropout(dropout)

        self.residual = Residual(
            in_dim = in_dim,
            residual_in_dim = residual_in_dim,
            mode = residual_type,
        )

    def forward(
        self,
        input: Tensor,
        residual_input: Tensor,
    ) -> Tensor:
        if self.residual_position == 'before':
            output = self.residual(input=input, residual_input=residual_input)
            output = self.batch_norm(output)
            output = self.activation(output)
            output = self.dropout(output)
        elif self.residual_position == 'after':
            output = self.batch_norm(input)
            output = self.activation(output)
            output = self.dropout(output)
            output = self.residual(input=output, residual_input=residual_input)
        else:
            raise ValueError

        return output


class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        batch_norm: bool = False,
        activation: str = 'none',
        dropout: float = 0.,
    ):
        super().__init__()
        
        self.in_dim = in_dim 
        self.out_dim = out_dim
        
        self.sequential = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=bias),
            BatchNorm(in_dim=out_dim, enable=batch_norm),
            Activation(activation),
            nn.Dropout(dropout),
        )
        
    def forward(
        self, 
        input: Tensor,
    ) -> Tensor:
        return self.sequential(input)
