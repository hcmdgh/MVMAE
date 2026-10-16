import torch 
import torch.nn.functional as F
from torch import Tensor 


def compute_reconstruction_loss(
    input: Tensor,
    target: Tensor,
    loss_func: str,
    alpha: float,
) -> Tensor:
    batch_size, sequence_len, embedding_dim = input.shape 
    assert input.shape == target.shape 
    
    if loss_func == 'cosine':
        input = F.normalize(input, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        loss = (1. - (input * target).sum(dim=-1)).pow(alpha)
        assert loss.shape == (batch_size, sequence_len)

        loss = loss.mean()
    elif loss_func == 'mse':
        loss = F.mse_loss(input=input, target=target)
    else:
        raise ValueError 
    
    return loss 
