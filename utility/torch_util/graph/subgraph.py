from ..common import * 


def node_subgraph(nids: Any,
                  edge_index: EdgeIndex,
                  num_nodes: Optional[int] = None) -> tuple[BoolTensor, BoolTensor, EdgeIndex, EdgeIndex]:
    device = edge_index.device
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1])

    if not isinstance(nids, Tensor):
        nids = torch.tensor(nids, dtype=torch.int64, device=device)
    assert nids.ndim == 1 

    if nids.dtype == torch.bool: 
        node_mask = nids 
        num_nodes = len(node_mask) 
    elif nids.dtype == torch.int64:
        assert num_nodes  
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device) 
        node_mask[nids] = True 
    else:
        raise TypeError 

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    sub_edge_index = edge_index[:, edge_mask]

    node_idx = torch.zeros(num_nodes, dtype=torch.int64, device=device)
    node_idx[node_mask] = torch.arange(int(node_mask.sum()), device=device)
    relabeled_sub_edge_index = node_idx[sub_edge_index]
    
    assert node_mask.shape == (num_nodes,) 
    assert edge_mask.shape == (num_edges,) 
    assert sub_edge_index.shape == relabeled_sub_edge_index.shape == (2, int(edge_mask.sum()))

    return node_mask, edge_mask, sub_edge_index, relabeled_sub_edge_index 
