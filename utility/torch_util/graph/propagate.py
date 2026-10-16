from ..common import * 

from .sparse import row_normalize_sparse_matrix,convert_edge_index_to_sparse_matrix


def propagate_node_feature(adj_mat: dglsp.SparseMatrix, 
                           node_feat: FloatTensor, 
                           num_hops: int,
                           use_tqdm: bool = False,
                           include_self: bool = True) -> FloatTensor:
    adj_mat = adj_mat.coalesce()
    
    num_nodes, node_feat_dim = node_feat.shape 
    
    prop_feat = node_feat 
    prop_feat_list = [] 
    
    if include_self:
        prop_feat_list.append(node_feat)

    A = adj_mat.transpose() 
    A = row_normalize_sparse_matrix(A) 
        
    for i in tqdm(range(num_hops), disable=not use_tqdm):   
        prop_feat = A @ prop_feat 
        
        prop_feat_list.append(prop_feat)   

    prop_feat_seq = torch.stack(prop_feat_list, dim=1) 

    if include_self:
        assert prop_feat_seq.shape == (num_nodes, num_hops + 1, node_feat_dim) 
    else:
        assert prop_feat_seq.shape == (num_nodes, num_hops, node_feat_dim) 

    return prop_feat_seq 

def mask_edge(edge_index: Tensor, p: float=0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')    
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]

def mask_propagate_node_feature(edge_index: EdgeIndex, 
                           node_feat: FloatTensor, 
                           num_hops: int,
                           mask_ratio: float=0.,
                           use_tqdm: bool = False,
                           include_self: bool = True) -> FloatTensor:
       
    num_nodes, node_feat_dim = node_feat.shape 
    
    prop_feat = node_feat 
    prop_feat_list = [] 
    
    if include_self:
        prop_feat_list.append(node_feat)
    
    for i in tqdm(range(num_hops), disable=not use_tqdm): 
        remaining_edges, _ = mask_edge(edge_index, mask_ratio)
        adj_mat = convert_edge_index_to_sparse_matrix(edge_index=remaining_edges, num_nodes=len(node_feat))
        adj_mat = adj_mat.coalesce()
        A = adj_mat.transpose() 
        A = row_normalize_sparse_matrix(A)   
        prop_feat = A @ prop_feat 
        
        prop_feat_list.append(prop_feat)   

    prop_feat_seq = torch.stack(prop_feat_list, dim=1) 

    if include_self:
        assert prop_feat_seq.shape == (num_nodes, num_hops + 1, node_feat_dim) 
    else:
        assert prop_feat_seq.shape == (num_nodes, num_hops, node_feat_dim) 

    return prop_feat_seq 