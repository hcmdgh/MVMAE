from ..common import * 


def compute_acc(pred: Any,
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
        pred = torch.argmax(pred, dim=-1)  
        assert pred.shape == (N,) 
        
        acc = torch.mean((pred == target).to(torch.float32)) 
        
        return float(acc) 
    
    elif target.shape == (N, C) and target.dtype == torch.bool: 
        _pred = torch.zeros(N, C, dtype=torch.bool, device=pred.device)   
        _pred[pred >= 0] = True 
        
        same = torch.all(_pred == target, dim=-1).to(torch.float32) 
        assert same.shape == (N,) 
        
        acc = torch.mean(same) 
        
        return float(acc)  
    
    else:
        raise TypeError 
