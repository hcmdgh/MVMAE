import torch
from torch import Tensor 
from cuml.cluster import KMeans as KMeans_cuml
from sklearn.cluster import KMeans as KMeans_sklearn


class KMeans:
    def __init__(
        self,
        num_clusters: int,
        init: str = 'k-means++',
        init_cnt: int = 3,
        seed: int = 42,
    ):
        self.num_clusters = num_clusters

        self.model = KMeans_cuml(
            n_clusters = num_clusters,
            init = init,
            n_init = init_cnt,
            random_state = seed,
        )

    def fit_predict(
        self, 
        input: Tensor,
    ) -> tuple[Tensor, Tensor]:
        device = input.device
        batch_size, in_dim = input.shape

        cluster = self.model.fit_predict(input)
        cluster = torch.tensor(cluster, dtype=torch.long, device=device)
        assert cluster.shape == (batch_size,) 

        center = self.model.cluster_centers_
        center = torch.tensor(center, dtype=torch.float, device=device)
        assert center.shape == (self.num_clusters, in_dim)

        return cluster, center
