import torch 
from torch import Tensor


def create_random_mask(
    size: int,
    mask_ratio: float,
) -> Tensor:
    mask_cnt = int(size * mask_ratio)
    mask_cnt = max(mask_cnt, 1)
    mask_cnt = min(mask_cnt, size - 1)

    perm = torch.randperm(size) 
    masked_index = perm[:mask_cnt]
    assert masked_index.shape == (mask_cnt,)

    mask = torch.zeros(size, dtype=torch.bool)
    mask[masked_index] = True 
        
    return mask 
