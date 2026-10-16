from torch_cluster import random_walk as _random_walk 

from ..common import * 


def random_walk(edge_index: EdgeIndex, 
                num_nodes: int, 
                seed_nids: IntTensor,
                walk_length: int,
                revisit_prob: float = 1.) -> IntTensor: 
    seed_cnt, = seed_nids.shape 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    
    out = _random_walk(
        row = edge_index[0], 
        col = edge_index[1], 
        num_nodes = num_nodes, 
        start = seed_nids, 
        walk_length = walk_length, 
        p = revisit_prob, 
    )
    assert out.shape == (seed_cnt, walk_length + 1) 
    
    return out 
