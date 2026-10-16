from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer 

from ..common import * 


def create_cosine_scheduler(optimizer: Optimizer, 
                            num_epochs: int) -> LambdaLR: 
    lr_fn = lambda epoch: ( 1 + np.cos( epoch * np.pi / num_epochs) ) * 0.5

    scheduler = LambdaLR(
        optimizer = optimizer, 
        lr_lambda = lr_fn, 
    )

    return scheduler 
