from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

from ..common import * 


def perform_KMeans(X: FloatArrayTensor, 
                   y: IntArrayTensor,
                   num_clusters: int = -1, 
                   repeat_cnt: int = 10) -> list[IntArray]:
    if isinstance(X, Tensor):
        X = X.detach().cpu().numpy() 
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy() 
        
    N, D = X.shape 
    assert y.shape == (N,)
    
    if num_clusters <= 0:
        num_clusters = int(np.max(y)) + 1 
        
    y_pred_list = [] 

    for i in range(repeat_cnt):
        seed = random.randrange(0, 10000)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')

        y_pred = kmeans.fit_predict(X) 
        assert y_pred.shape == (N,)

        y_pred_list.append(y_pred) 
        
    return y_pred_list


def evaluate_KMeans(X: FloatArrayTensor, 
                    y: IntArrayTensor,
                    num_clusters: int = -1, 
                    repeat_cnt: int = 10) -> tuple[float, float]:
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy() 

    y_pred_list = perform_KMeans(
        X = X, 
        y = y, 
        num_clusters = num_clusters, 
        repeat_cnt = repeat_cnt, 
    )
    
    nmi_list = []
    ari_list = []
    
    for y_pred in y_pred_list:
        nmi = normalized_mutual_info_score(labels_true=y, labels_pred=y_pred)
        ari = adjusted_rand_score(labels_true=y, labels_pred=y_pred)

        nmi_list.append(nmi) 
        ari_list.append(ari) 
        
    nmi = float(np.mean(nmi_list))
    ari = float(np.mean(ari_list))
    
    return nmi, ari 
