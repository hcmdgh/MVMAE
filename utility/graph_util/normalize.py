import torch 
from torch import Tensor 


def normalize_feature(
    feat: Tensor,
) -> Tensor:
    batch_size, feat_dim = feat.shape 
    assert feat.dtype == torch.float32 
    
    mean = feat.mean(dim=0, keepdim=True)
    assert mean.shape == (1, feat_dim)

    std = feat.std(dim=0, keepdim=True, unbiased=False)
    assert std.shape == (1, feat_dim)

    output = (feat - mean) / (std + 1e-6)
    assert output.shape == (batch_size, feat_dim)
    
    return output
