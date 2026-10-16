from ..common import * 


def degree(index: IntTensor, 
           num_nodes: int) -> IntTensor: 
    device = index.device 
    num_edges, = index.shape 
    
    out = torch.zeros(num_nodes, dtype=torch.int64, device=device)
    ones = torch.ones(num_edges, dtype=torch.int64, device=device) 
    
    out.scatter_add_(
        dim = 0, 
        index = index, 
        src = ones, 
    )
    
    return out 


def in_degree(edge_index: EdgeIndex, 
              num_nodes: int) -> IntTensor: 
    return degree(index=edge_index[1], num_nodes=num_nodes)


def out_degree(edge_index: EdgeIndex, 
               num_nodes: int) -> IntTensor: 
    return degree(index=edge_index[0], num_nodes=num_nodes)
