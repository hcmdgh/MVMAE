from ..common import * 


def add_self_loop(edge_index: EdgeIndex,
                  num_nodes: int) -> EdgeIndex:
    device = edge_index.device 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
                  
    mask = edge_index[0] != edge_index[1]

    loop_index = torch.arange(num_nodes, dtype=torch.int64, device=device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1) 
    assert loop_index.shape == (2, num_nodes) 

    new_edge_index = torch.cat([edge_index[:, mask], loop_index], dim=-1)

    return new_edge_index 


def remove_self_loop(edge_index: EdgeIndex) -> EdgeIndex:
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 

    mask = edge_index[0] != edge_index[1]
    new_edge_index = edge_index[:, mask] 
    
    return new_edge_index 


def remove_self_loop_with_edge_weight(edge_index: EdgeIndex,
                                      edge_weight: FloatTensor) -> tuple[EdgeIndex, FloatTensor]:
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    assert edge_weight.shape == (num_edges,) 

    mask = edge_index[0] != edge_index[1]
    new_edge_index = edge_index[:, mask] 
    new_edge_weight = edge_weight[mask]
    
    return new_edge_index, new_edge_weight 
