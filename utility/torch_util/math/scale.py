from sklearn.preprocessing import StandardScaler

from ..common import * 


def standardize_scale_feature(feat: FloatTensor) -> FloatTensor: 
    device = feat.device 
    N, D = feat.shape 
    
    feat = feat.cpu().numpy() 
    
    scaler = StandardScaler()
    scaler.fit(feat)
    out = scaler.transform(feat) 
    
    out = torch.tensor(out, dtype=torch.float32, device=device) 
    assert out.shape == (N, D) 
    
    return out 
