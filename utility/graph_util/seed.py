import random 
import torch 
import numpy as np 


def set_seed(
    seed: int,
    deterministic: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"{seed = }, {deterministic = }\n")
