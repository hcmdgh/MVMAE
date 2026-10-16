from itertools import zip_longest 

from ..common import * 


def ema_update(src_model: nn.Module, 
               target_model: nn.Module,
               momentum: float): 
    with torch.no_grad():
        for src_param, target_param in zip_longest(src_model.parameters(), target_model.parameters()):
            target_param.data = target_param.data * momentum + src_param.data * (1. - momentum) 
