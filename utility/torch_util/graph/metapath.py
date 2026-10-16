from ..common import * 
from ..util.device import to_device, get_device
from .sparse import convert_edge_index_to_coo_matrix
from .etype import to_canonical_etype
from .sparse import row_normalize_sparse_matrix


def metapath_reachable_graph(edge_index_dict: dict[EdgeType, EdgeIndex],
                             num_nodes_dict: dict[NodeType, int],
                             metapath: list[str]) -> EdgeIndex:
    device = next(iter(edge_index_dict.values())).device 
    _edge_index_dict = { etype: tuple(edge_index) for etype, edge_index in edge_index_dict.items() } 
    
    hg = dgl.heterograph(_edge_index_dict, num_nodes_dict=num_nodes_dict) 
    
    out_g = dgl.metapath_reachable_graph(g=hg, metapath=metapath)

    out_edge_index = torch.stack(out_g.edges()).to(device) 
    
    return out_edge_index 


def metapath_reachable_graph_old(edge_index_dict: dict[EdgeType, EdgeIndex],
                             num_nodes_dict: dict[NodeType, int],
                             metapath: list[str]) -> tuple[EdgeIndex, int, int]:
    raise NotImplementedError 

    device = next(iter(edge_index_dict.values())).device 
    
    A_accum = None 
                             
    for etype in metapath: 
        etype = to_canonical_etype(etype=etype, etypes=edge_index_dict)
        src_ntype, _, dest_ntype = etype 
        num_src_nodes, num_dest_nodes = num_nodes_dict[src_ntype], num_nodes_dict[dest_ntype] 
        edge_index = edge_index_dict[etype] 
        
        A = convert_edge_index_to_coo_matrix(
            edge_index = edge_index, 
            num_nodes = (num_src_nodes, num_dest_nodes), 
        ) 
        
        if A_accum is None: 
            A_accum = A 
        else:
            A_accum = A_accum @ A 
            
    assert A_accum is not None 

    num_src_nodes, num_dest_nodes = A_accum.shape 
    
    edge_index = np.stack(A_accum.nonzero()) 
    edge_index = torch.tensor(edge_index, dtype=torch.int64, device=device) 
     
    return edge_index, num_src_nodes, num_dest_nodes 


def propagate_metapath_feature(adj_mat_dict: dict[EdgeType, dglsp.SparseMatrix], 
                               node_feat_dict: dict[NodeType, FloatTensor], 
                               metapath_list: list[list[str]], 
                               use_tqdm: bool = False,
                               use_gpu: bool = True) -> FloatTensor:
    if use_gpu: 
        adj_mat_dict = to_device(adj_mat_dict)
        node_feat_dict = to_device(node_feat_dict) 
    else:
        adj_mat_dict = to_device(adj_mat_dict, device='cpu')
        node_feat_dict = to_device(node_feat_dict, device='cpu') 
                               
    aggr_feat_list = [] 

    for metapath in tqdm(metapath_list, disable=not use_tqdm, desc='Propagating'):
        if len(metapath) == 1 and metapath[0] in node_feat_dict.keys():
            aggr_feat = node_feat_dict[metapath[0]]
        else:
            src_etype = to_canonical_etype(etype=metapath[0], etypes=adj_mat_dict.keys()) 
            src_feat = node_feat_dict[src_etype[0]]
            aggr_feat = src_feat 
                
            for etype in metapath: 
                etype = to_canonical_etype(etype=etype, etypes=adj_mat_dict.keys()) 
                
                A = adj_mat_dict[etype]  
                A = A.transpose() 
                A = row_normalize_sparse_matrix(A) 
                
                aggr_feat = A @ aggr_feat  

        aggr_feat_list.append(aggr_feat.cpu())  
        
    aggr_feat_seq = torch.stack(aggr_feat_list, dim=1) 
    assert aggr_feat_seq.ndim == 3  

    return aggr_feat_seq 


def propagate_metapath_feature_without_self(adj_mat_dict: dict[EdgeType, dglsp.SparseMatrix],
                                            metapath: list[str], 
                                            src_feat: FloatTensor, 
                                            use_gpu: bool = True) -> FloatTensor:
    raise NotImplementedError
    A_accum = compute_metapath_adjacency_matrix(
        adj_mat_dict = adj_mat_dict, 
        metapath = metapath, 
        use_gpu = use_gpu, 
    )
    
    A_accum = remove_diag(A_accum) 
    
    aggr_feat = A_accum @ src_feat 
        
    return aggr_feat 


def compute_metapath_adjacency_matrix(adj_mat_dict: dict[EdgeType, dglsp.SparseMatrix], 
                                      metapath: list[str], 
                                      use_gpu: bool = True) -> dglsp.SparseMatrix:
    if use_gpu:
        adj_mat_dict = to_device(adj_mat_dict) 
    else: 
        adj_mat_dict = to_device(adj_mat_dict, device='cpu') 
    
    A_prod = None 
        
    for etype in metapath: 
        etype = to_canonical_etype(etype=etype, etypes=adj_mat_dict.keys()) 
        
        A = adj_mat_dict[etype]  
        A = A.transpose() 
        A = row_normalize_sparse_matrix(A) 
        
        if A_prod is None: 
            A_prod = A 
        else: 
            A_prod = A @ A_prod  

    assert A_prod is not None 

    return A_prod 


def metapath_reachable_subgraph(seed_node_idx: Any, 
                                adj_list_dict: dict[EdgeType, dict[int, set[int]]],
                                metapath: list[str]) -> dict[int, set[int]]:
    if isinstance(seed_node_idx, Tensor):
        seed_node_idx = set(seed_node_idx.tolist()) 
    elif isinstance(seed_node_idx, (list, set)): 
        seed_node_idx = set(seed_node_idx) 
    else:
        raise TypeError 
    

def to_canonical_metapath(metapath: list[str],
                          etypes: Iterable[EdgeType]) -> list[EdgeType]:
    _metapath = []

    for etype in metapath:
        _etype = to_canonical_etype(etype=etype, etypes=etypes)
        _metapath.append(_etype) 
        
    return _metapath
