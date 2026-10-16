from ..common import * 


def compute_hits_at_k(pos_score: FloatTensor,
                      neg_score: FloatTensor,
                      k: int) -> float:
    """
    计算Hits@K。
    
    给定每个正样本和负样本的分数，
    设正样本数量为P，负样本数量为N，
    对于每个正样本p，将其与所有负样本一起排名，
    如果p在排名中位于前K，则记为1，否则记为0，
    这样可以得到长度为P的0/1数组，
    对其求均值可得Hits@K的值。
    """
    
    pos_score = pos_score.detach() 
    neg_score = neg_score.detach() 
    assert pos_score.ndim == neg_score.ndim == 1 
    
    if len(neg_score) < k:
        return 1. 

    kth_score_in_neg = torch.topk(neg_score, k)[0][-1]
    
    result = float( torch.mean( (pos_score > kth_score_in_neg).float() ) ) 
    
    return result 
