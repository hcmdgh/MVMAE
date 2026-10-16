from ..common import * 


def compute_cosine_loss(h1: FloatTensor, 
                        h2: FloatTensor) -> FloatScalarTensor:
    N, D = h1.shape 
    assert h1.shape == h2.shape == (N, D) 
                                
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    loss = (1. - (h1 * h2).sum(dim=-1))
    loss = loss.mean()

    return loss


def compute_scaled_cosine_loss(h1: FloatTensor, 
                               h2: FloatTensor, 
                               alpha: float = 1.) -> FloatScalarTensor:
    N, D = h1.shape 
    assert h1.shape == h2.shape == (N, D) 
                                
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    loss = (1. - (h1 * h2).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()

    return loss
