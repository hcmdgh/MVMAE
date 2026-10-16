from ..common import * 


def split_train_val_test_set(total_cnt: int,
                             train_ratio: float,
                             val_ratio: float) -> tuple[BoolArray, BoolArray, BoolArray]:
    raise NotImplementedError
    test_ratio = 1. - train_ratio - val_ratio 
    assert test_ratio > 0. 
    
    train_cnt = int(total_cnt * train_ratio)
    val_cnt = int(total_cnt * val_ratio)
    train_mask = np.zeros(total_cnt, dtype=bool)
    val_mask = np.zeros(total_cnt, dtype=bool)
    test_mask = np.zeros(total_cnt, dtype=bool)
    perm = np.random.permutation(total_cnt)
    train_mask[perm[:train_cnt]] = True 
    val_mask[perm[train_cnt: train_cnt + val_cnt]] = True 
    test_mask[perm[train_cnt + val_cnt:]] = True 
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == total_cnt 
    
    return train_mask, val_mask, test_mask 


def split_train_val_set(full_set: Any,
                        train_ratio: float) -> tuple[BoolTensor, BoolTensor]: 
    raise NotImplementedError
    if isinstance(full_set, int):
        device = get_device()
        full_mask = torch.ones(full_set, dtype=torch.bool, device=device)  
    elif isinstance(full_set, np.ndarray) and full_set.dtype == bool: 
        device = get_device()
        full_mask = torch.tensor(full_set, dtype=torch.bool, device=device) 
    elif isinstance(full_set, Tensor) and full_set.dtype == torch.bool: 
        device = full_set.device 
        full_mask = full_set  
    else:
        raise TypeError 
    
    N, = full_mask.shape 
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device) 

    full_idx = full_mask.nonzero().view(-1)  
    M, = full_idx.shape 
    train_cnt = int(M * train_ratio) 
    
    perm = np.random.permutation(M) 
    train_mask[full_idx[perm[:train_cnt]]] = True 
    val_mask[full_idx[perm[train_cnt:]]] = True 
    
    assert train_mask.sum() + val_mask.sum() == M 
    assert torch.all(~(train_mask & val_mask)) 
    assert torch.all(train_mask | val_mask == full_mask)

    return train_mask, val_mask 


def random_split_train_val_test_set(n: int,
                                    train_ratio: float, 
                                    val_ratio: float,
                                    output_format: str = 'mask') -> tuple[Tensor, Tensor, Tensor]:
    train_cnt = int(n * train_ratio) 
    val_cnt = int(n * val_ratio) 
    test_cnt = n - train_cnt - val_cnt 
    assert train_cnt > 0 and val_cnt > 0 and test_cnt > 0 
    
    perm = torch.randperm(n, dtype=torch.int64) 
    
    train_idx = perm[:train_cnt] 
    val_idx = perm[train_cnt: train_cnt + val_cnt] 
    test_idx = perm[train_cnt + val_cnt:] 
    assert len(train_idx) + len(val_idx) + len(test_idx) == n 
    
    if output_format == 'index':
        return train_idx, val_idx, test_idx
    elif output_format == 'mask':
        train_mask = torch.zeros(n, dtype=torch.bool) 
        val_mask = torch.zeros(n, dtype=torch.bool) 
        test_mask = torch.zeros(n, dtype=torch.bool) 
        train_mask[train_idx] = True  
        val_mask[val_idx] = True  
        test_mask[test_idx] = True  
        
        return train_mask, val_mask, test_mask
    else:
        raise AssertionError
    
    
def random_split_train_test_set(n: int,
                                train_ratio: float, 
                                output_format: str = 'mask') -> tuple[ndarray, ndarray]:
    train_cnt = int(n * train_ratio) 
    test_cnt = n - train_cnt 
    assert train_cnt > 0 and test_cnt > 0 
    
    perm = np.random.permutation(n) 
    
    train_idx = perm[:train_cnt] 
    test_idx = perm[train_cnt:] 
    assert len(train_idx) + len(test_idx) == n 
    
    if output_format == 'index':
        return train_idx, test_idx
    elif output_format == 'mask':
        train_mask = np.zeros(n, dtype=bool) 
        test_mask = np.zeros(n, dtype=bool) 
        train_mask[train_idx] = True  
        test_mask[test_idx] = True  
        
        return train_mask, test_mask
    else:
        raise AssertionError


def random_split_edge_index(edge_index: EdgeIndex,
                            test_ratio: float,
                            is_undirected: bool = True) -> tuple[EdgeIndex, EdgeIndex]:
    device = edge_index.device 
    num_edges = edge_index.shape[-1] 
    assert edge_index.shape == (2, num_edges) 
    assert 0. < test_ratio < 1. 
    assert is_undirected 
    
    edge_set: set[tuple[int, int]] = set() 
    
    for src_nid, dest_nid in zip(*edge_index.tolist()): 
        if src_nid > dest_nid: 
            src_nid, dest_nid = dest_nid, src_nid 
            
        edge_set.add((src_nid, dest_nid)) 
        
    E = len(edge_set) 
    edge_index = torch.tensor(list(edge_set), dtype=torch.int64, device=device).T 
    assert edge_index.shape == (2, E) 
    perm = np.random.permutation(E) 
    test_cnt = int(E * test_ratio) 
    test_edge_index = edge_index[:, perm[:test_cnt]]
    train_edge_index = edge_index[:, perm[test_cnt:]] 
    assert test_edge_index.shape[-1] + train_edge_index.shape[-1] == E 
    
    test_edge_index = torch.cat([test_edge_index, torch.flip(test_edge_index, dims=[0])], dim=-1)
    train_edge_index = torch.cat([train_edge_index, torch.flip(train_edge_index, dims=[0])], dim=-1)
    assert test_edge_index.shape[-1] + train_edge_index.shape[-1] == E * 2 

    return train_edge_index, test_edge_index 
