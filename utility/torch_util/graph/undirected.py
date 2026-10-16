from ..common import * 


def to_undirected(edge_index: EdgeIndex) -> EdgeIndex:
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    
    row, col = edge_index 
    row, col = torch.cat([row, col]), torch.cat([col, row])
    new_edge_index = torch.stack([row, col]) 

    new_edge_index = torch.unique(new_edge_index, dim=-1)
    
    return new_edge_index 


def add_reverse_edges(edge_index: EdgeIndex) -> EdgeIndex: 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    
    row, col = edge_index 
    row, col = torch.cat([row, col]), torch.cat([col, row])
    new_edge_index = torch.stack([row, col]) 
    
    return new_edge_index 
