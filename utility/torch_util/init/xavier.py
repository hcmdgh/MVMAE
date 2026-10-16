from ..common import * 


def xavier_uniform_(obj: Any): 
    if isinstance(obj, FloatTensor): 
        if obj.ndim > 1:
            nn.init.xavier_uniform_(obj) 
        else:
            nn.init.xavier_uniform_(obj.view(1, -1)) 
    else:
        raise TypeError 
