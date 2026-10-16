from torch_geometric.nn import MetaPath2Vec as _MetaPath2Vec 

from ..imports import * 
from ..hetero_graph import to_canonical_etype 

__all__ = [
    'Metapath2Vec_PyG', 
]


class Metapath2Vec_PyG(nn.Module):
    def __init__(self,
                 edge_index_dict: dict[EdgeType, EdgeIndex], 
                 num_nodes_dict: dict[NodeType, int],
                 embed_dim: int, 
                 metapath: list[str], 
                 walk_length: int, 
                 context_size: int,
                 walks_per_node: int, 
                 num_negative_samples: int,
                 sparse: bool = False):
        super().__init__()
        
        _metapath = [
            to_canonical_etype(etype=etype, etypes=edge_index_dict.keys()) 
            for etype in metapath 
        ] 
        
        self.model = _MetaPath2Vec(
            edge_index_dict = edge_index_dict, 
            num_nodes_dict = num_nodes_dict, 
            embedding_dim = embed_dim, 
            metapath = _metapath, 
            walk_length = walk_length, 
            context_size = context_size, 
            walks_per_node = walks_per_node, 
            num_negative_samples = num_negative_samples, 
            sparse = sparse, 
        )
        
    def get_loader(self,
                   batch_size: int,
                   shuffle: bool = True) -> DataLoader:
        loader = self.model.loader(
            batch_size = batch_size, 
            shuffle = shuffle, 
        )
        
        return loader 

    def compute_loss(self,
                     pos_rw: FloatTensor, 
                     neg_rw: FloatTensor) -> FloatScalarTensor: 
        return self.model.loss(
            pos_rw = pos_rw, 
            neg_rw = neg_rw, 
        )

    @torch.no_grad() 
    def compute_embedding(self,
                          ntype: NodeType) -> FloatTensor: 
        emb = self.model.forward(node_type=ntype).detach() 
        
        return emb 
