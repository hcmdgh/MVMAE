from ..common import * 


def compute_cosine_similarity_matrix(h1: FloatTensor, 
                                     h2: FloatTensor) -> FloatTensor:
    N, D = h1.shape 
    assert h1.shape == h2.shape == (N, D) 
                                
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    out = h1 @ h2.T 
    assert out.shape == (N, N)

    return out 
