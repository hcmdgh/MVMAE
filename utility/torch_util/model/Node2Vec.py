from ..imports import * 
from ..graph import random_walk

__all__ = [
    'Node2Vec', 
]


class Node2Vec(nn.Module):
    def __init__(self,
                 edge_index: EdgeIndex, 
                 num_nodes: int, 
                 walk_length: int,
                 context_size: int,
                 embedding_dim: int, 
                 walks_per_node: int = 1,
                 num_negative_samples: int = 1,
                 revisit_prob: float = 1.):
        super().__init__() 
        
        self.edge_index = edge_index 
        self.num_nodes = num_nodes 
        assert walk_length >= context_size 
        self.walk_length = walk_length - 1 
        self.context_size = context_size
        self.walks_per_node = walks_per_node 
        self.num_negative_samples = num_negative_samples 
        self.revisit_prob = revisit_prob 
        self.embedding_dim = embedding_dim
        self.device = self.edge_index.device 
        
        self.embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim) 

    @torch.no_grad() 
    def get_embedding(self,
                      mask: BoolTensor) -> FloatTensor: 
        indices = mask.nonzero().squeeze() 
        assert indices.ndim == 1 
        
        emb = self.embedding(indices) 
        
        return emb 
    
    def positive_sample(self, 
                        seed_nids: IntTensor) -> IntTensor:
        seed_cnt, = seed_nids.shape 
        
        seed_nids = seed_nids.repeat(self.walks_per_node)
        assert seed_nids.shape == (seed_cnt * self.walks_per_node,) 
        
        path_mat = random_walk(
            edge_index = self.edge_index, 
            num_nodes = self.num_nodes, 
            seed_nids = seed_nids, 
            walk_length = self.walk_length, 
            revisit_prob = self.revisit_prob, 
        )
        assert path_mat.shape == (seed_cnt * self.walks_per_node, self.walk_length + 1) 
        
        path_list = [
            path_mat[:, j: j + self.context_size]
            for j in range(self.walk_length + 1 + 1 - self.context_size)
        ] 
        
        path_mat = torch.cat(path_list, dim=0) 
        assert path_mat.shape == (seed_cnt * self.walks_per_node * (self.walk_length + 1 + 1 - self.context_size), self.context_size)
                
        return path_mat 
    
    def negative_sample(self, 
                        seed_nids: IntTensor) -> IntTensor:
        seed_cnt, = seed_nids.shape
        
        seed_nids = seed_nids.repeat(self.walks_per_node * self.num_negative_samples)
        assert seed_nids.shape == (seed_cnt * self.walks_per_node * self.num_negative_samples,) 

        path_mat = torch.randint(low=0, high=self.num_nodes, size=[len(seed_nids), self.walk_length], device=self.device) 
        path_mat = torch.cat([seed_nids.view(-1, 1), path_mat], dim=-1)
        assert path_mat.shape == (seed_cnt * self.walks_per_node * self.num_negative_samples, self.walk_length + 1) 

        path_list = [
            path_mat[:, j: j + self.context_size]
            for j in range(self.walk_length + 1 + 1 - self.context_size)
        ] 
        
        path_mat = torch.cat(path_list, dim=0) 
        assert path_mat.shape == (seed_cnt * self.walks_per_node * self.num_negative_samples * (self.walk_length + 1 + 1 - self.context_size), self.context_size)
            
        return path_mat 

    def sample(self, 
               seed_nids: Any)  -> tuple[IntTensor, IntTensor]: 
        if not isinstance(seed_nids, Tensor): 
            seed_nids = torch.tensor(seed_nids, dtype=torch.int64, device=self.device) 
               
        return self.positive_sample(seed_nids), self.negative_sample(seed_nids) 
    
    def compute_loss(self, 
                     pos_path_mat: IntTensor,
                     neg_path_mat: IntTensor) -> FloatScalarTensor: 
        pos_walk_cnt = len(pos_path_mat) 
        neg_walk_cnt = len(neg_path_mat) 
        assert pos_path_mat.shape == (pos_walk_cnt, self.context_size)
        assert neg_path_mat.shape == (neg_walk_cnt, self.context_size)

        start = pos_path_mat[:, :1] 
        assert start.shape == (pos_walk_cnt, 1) 
        
        rest = pos_path_mat[:, 1:] 
        assert rest.shape == (pos_walk_cnt, self.context_size - 1)
        
        h_start = self.embedding(start) 
        assert h_start.shape == (pos_walk_cnt, 1, self.embedding_dim)
        
        h_rest = self.embedding(rest) 
        assert h_rest.shape == (pos_walk_cnt, self.context_size - 1, self.embedding_dim) 
        
        pos_pred = (h_start * h_rest).sum(-1).view(-1) 
        assert pos_pred.shape == (pos_walk_cnt * (self.context_size - 1),)

        pos_loss = F.binary_cross_entropy_with_logits(input=pos_pred, target=torch.ones_like(pos_pred))
        
        start = neg_path_mat[:, :1] 
        assert start.shape == (neg_walk_cnt, 1) 
        
        rest = neg_path_mat[:, 1:] 
        assert rest.shape == (neg_walk_cnt, self.context_size - 1)
        
        h_start = self.embedding(start) 
        assert h_start.shape == (neg_walk_cnt, 1, self.embedding_dim)
        
        h_rest = self.embedding(rest) 
        assert h_rest.shape == (neg_walk_cnt, self.context_size - 1, self.embedding_dim) 
        
        neg_pred = (h_start * h_rest).sum(-1).view(-1) 
        assert neg_pred.shape == (neg_walk_cnt * (self.context_size - 1),)
        
        neg_loss = F.binary_cross_entropy_with_logits(input=neg_pred, target=torch.zeros_like(neg_pred))
        
        loss = pos_loss + neg_loss 
        
        return loss 
