from ..imports import * 
from ..graph import node_subgraph, random_walk

__all__ = [
    'GraphSAINTSampler', 
    'GraphSAINTNodeSampler', 
    'GraphSAINTEdgeSampler', 
    'GraphSAINTRandomWalkSampler', 
]


class GraphSAINTSampler:
    def __init__(self,
                 edge_index: EdgeIndex,
                 num_nodes: int,
                 batch_size: int,
                 num_steps: int,
                 sample_coverage: int = 0):
        self.device = edge_index.device 
        self.edge_index = edge_index 
        self.num_nodes = num_nodes 
        self.num_edges = self.edge_index.shape[-1]
        assert self.edge_index.shape == (2, self.num_edges) 
        self.batch_size = batch_size  
        self.num_steps = num_steps 
        
        if sample_coverage > 0:
            raise NotImplementedError 
        
    def __len__(self) -> int: 
        return self.num_steps 
    
    def sample_nodes(self) -> IntTensor:
        raise NotImplementedError 
    
    def __iter__(self) -> Iterable[tuple[BoolTensor, BoolTensor, EdgeIndex, EdgeIndex]]:
        for step in range(self.num_steps): 
            node_idx = self.sample_nodes() 
            node_idx = node_idx.unique() 
            
            node_mask, edge_mask, sub_edge_index, relabeled_sub_edge_index = node_subgraph(
                node_idx, 
                edge_index = self.edge_index, 
                num_nodes = self.num_nodes, 
            )
            
            yield node_mask, edge_mask, sub_edge_index, relabeled_sub_edge_index


class GraphSAINTNodeSampler(GraphSAINTSampler):
    def sample_nodes(self) -> IntTensor:
        edge_idx = torch.randint(low=0, high=self.num_edges, size=[self.batch_size], device=self.device) 
        
        src_nids = self.edge_index[0, edge_idx] 
        assert src_nids.shape == (self.batch_size,) 
        
        return src_nids 
        
        
class GraphSAINTEdgeSampler(GraphSAINTSampler):
    pass 


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    def __init__(self, 
                 edge_index: EdgeIndex, 
                 num_nodes: int, 
                 batch_size: int, 
                 num_steps: int,
                 walk_length: int,  
                 sample_coverage: int = 0):
        super().__init__(
            edge_index = edge_index, 
            num_nodes = num_nodes, 
            batch_size = batch_size, 
            num_steps = num_steps, 
            sample_coverage = sample_coverage, 
        )

        self.walk_length = walk_length 
        
    def sample_nodes(self) -> IntTensor:
        seed_nids = torch.randint(low=0, high=self.num_nodes, size=[self.batch_size], device=self.device) 
        
        path_mat = random_walk(
            edge_index = self.edge_index, 
            num_nodes = self.num_nodes, 
            seed_nids = seed_nids, 
            walk_length = self.walk_length, 
        ) 
        
        node_idx = path_mat.view(-1) 
        
        return node_idx 
