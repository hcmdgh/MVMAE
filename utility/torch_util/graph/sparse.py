from ..common import * 


def row_normalize_sparse_matrix(A: dglsp.SparseMatrix) -> dglsp.SparseMatrix: 
    num_src_nodes, num_dest_nodes = A.shape 
    
    deg = A.sum(1) 
    assert deg.shape == (num_src_nodes,) 
    
    inv_deg = torch.pow(deg, -1)  
    inv_deg[torch.isinf(inv_deg)] = 0. 
    assert inv_deg.shape == (num_src_nodes,) 
    
    D = dglsp.diag(inv_deg) 
    assert D.shape == (num_src_nodes, num_src_nodes) 
    
    out = D @ A 
    assert out.shape == (num_src_nodes, num_dest_nodes) 
    
    return out 


def convert_edge_index_to_sparse_matrix(edge_index: EdgeIndex, 
                                        num_nodes: Union[int, tuple[int, int]]) -> dglsp.SparseMatrix: 
    if isinstance(num_nodes, int):
        num_src_nodes = num_dest_nodes = num_nodes 
    elif isinstance(num_nodes, (tuple, list)): 
        num_src_nodes, num_dest_nodes = num_nodes 
    else:
        raise TypeError
                                        
    sp_mat = dglsp.spmatrix(edge_index, shape=(num_src_nodes, num_dest_nodes)) 
    
    return sp_mat  
    
    
def convert_edge_index_dict_to_sparse_matrix_dict(edge_index_dict: dict[EdgeType, EdgeIndex], 
                                                  num_nodes_dict: dict[NodeType, int]) -> dict[EdgeType, dglsp.SparseMatrix]: 
    sp_mat_dict = {
        etype: convert_edge_index_to_sparse_matrix(
            edge_index = edge_index, 
            num_nodes = (num_nodes_dict[etype[0]], num_nodes_dict[etype[-1]]), 
        )
        for etype, edge_index in edge_index_dict.items() 
    }
    
    return sp_mat_dict 


def remove_diag(sp_mat: dglsp.SparseMatrix) -> dglsp.SparseMatrix:
    num_nodes = sp_mat.shape[0] 
    assert sp_mat.shape == (num_nodes, num_nodes)
    
    edge_index = torch.stack([sp_mat.row, sp_mat.col]) 
    edge_weight = sp_mat.val 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    assert edge_weight.shape == (num_edges,) 

    mask = edge_index[0] != edge_index[1]
    new_edge_index = edge_index[:, mask] 
    new_edge_weight = edge_weight[mask]

    new_sp_mat = dglsp.spmatrix(indices=new_edge_index, val=new_edge_weight, shape=(num_nodes, num_nodes))
    
    return new_sp_mat 


def convert_edge_index_to_coo_matrix(edge_index: EdgeIndex, 
                                     num_nodes: Union[int, tuple[int, int]]) -> sp.coo_matrix: 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1])
                                     
    row, col = edge_index.detach().cpu().numpy() 
    ones = np.ones(num_edges) 
    
    if isinstance(num_nodes, (tuple, list)): 
        shape = [num_nodes[0], num_nodes[1]] 
    elif isinstance(num_nodes, int): 
        shape = [num_nodes] 
    else:
        raise TypeError 
        
    coo_matrix = sp.coo_matrix((ones, (row, col)), shape=shape) 
    
    return coo_matrix 
