from ..common import * 


def compute_mrr(pred: Any,
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
        # argsort：第i个样本预测值最高的类别是argsort[i, 0] 
        argsort = torch.argsort(-pred, dim=-1) 
        assert argsort.shape == (N, C) 

        # mask：mask[i, j]为真表示第i个样本其真实标签的预测排名为j
        mask = (argsort == target.view(-1, 1))
        assert mask.shape == (N, C) 

        # rank：rank[i]表示第i个样本其真实标签的预测排名
        rank = torch.nonzero(mask)
        rank = rank[:, 1] + 1 
        assert rank.shape == (N,) 
        
        # mrr：对每个样本的真实标签的预测排名的倒数取平均值，得到MRR
        mrr = torch.mean(1. / rank) 
        
        return float(mrr) 

    elif target.shape == (N, C) and target.dtype == torch.bool: 
        # argsort：第i个样本预测值最高的类别是argsort[i, 0] 
        argsort = torch.argsort(-pred, dim=-1) 
        assert argsort.shape == (N, C) 

        # mask：mask[i, j]为真表示第i个样本其真实标签的预测排名为j
        mask = target[torch.arange(N).view(-1, 1), argsort] 
        assert mask.shape == (N, C) 

        # rank：rank[i]表示第i个样本其真实标签的预测排名（如果有多个真实标签，取最优排名）
        row, col = torch.nonzero(mask).T 
        col.add_(1) 
        rank = torch_scatter.scatter(src=col, index=row, dim_size=N, reduce='min')
        rank = rank.to(torch.float32) 
        assert rank.shape == (N,) 
        
        # mrr：对每个样本的真实标签的预测排名的倒数取平均值，得到MRR
        rank[rank == 0.] = torch.inf 
        mrr = torch.mean(1. / rank) 
        
        return float(mrr)

    else:
        raise TypeError
