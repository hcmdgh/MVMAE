from ..common import * 


def _compute_dcg_at_k(score_batch: FloatTensor,
                      k: int) -> FloatTensor: 
    device = score_batch.device 
    B, L = score_batch.shape  
    assert k > 0
    
    score_batch = score_batch[:, :k] 
    assert score_batch.shape == (B, k) 
    
    out = torch.sum( score_batch / ( torch.log2( torch.arange(2, k + 2, device=device) ) ).view(1, -1), dim=-1 ) 
    assert out.shape == (B,) 
    
    return out 
    
    
def _compute_ndcg_at_k(score_batch: FloatTensor,
                       k: int) -> FloatTensor: 
    B, L = score_batch.shape 
    assert k > 0

    sorted_score = torch.sort(score_batch, descending=True, dim=-1).values 
    idcg = _compute_dcg_at_k(score_batch=sorted_score, k=k) 
    
    dcg = _compute_dcg_at_k(score_batch=score_batch, k=k) 
    
    out = dcg / idcg 
    assert out.shape == (B,) 

    return out 


def compute_ndcg(pred: Any, 
                 target: Any) -> float:
    if isinstance(pred, ndarray): 
        pred = torch.from_numpy(pred)  
    if isinstance(target, ndarray): 
        target = torch.from_numpy(target) 
    assert isinstance(pred, Tensor) and isinstance(target, Tensor) 
    N, C = pred.shape 
    assert pred.dtype == torch.float32 
    pred, target = pred.detach(), target.detach() 
    
    if target.shape == (N,) and target.dtype == torch.int64:
        score_batch = ( torch.argsort(pred, descending=True, dim=-1) == target.view(-1, 1) ).float() 
        assert score_batch.shape == (N, C) 
        
        ndcg = _compute_ndcg_at_k(score_batch=score_batch, k=C) 
        assert ndcg.shape == (N,) 
        
        ndcg = float(torch.mean(ndcg))
        
        return ndcg  

    elif target.shape == (N, C) and target.dtype == torch.bool: 
        argsort = torch.argsort(pred, descending=True, dim=-1) 
        assert argsort.shape == (N, C) 
        
        score_batch = torch.gather(input=target, index=argsort, dim=-1).float() 
        assert score_batch.shape == (N, C) 
        
        ndcg = _compute_ndcg_at_k(score_batch=score_batch, k=C) 
        assert ndcg.shape == (N,) 
        
        ndcg = float(torch.mean(ndcg))
        
        return ndcg  

    else:
        raise TypeError
