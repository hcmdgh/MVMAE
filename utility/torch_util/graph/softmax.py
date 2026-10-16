from ..common import * 


def edge_softmax(edge_index: EdgeIndex, 
                 edge_feat: FloatTensor,
                 num_nodes: int, 
                 to_dest: bool = True) -> FloatTensor:
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    assert edge_feat.shape == (num_edges, feat_dim := edge_feat.shape[-1]) 
    
    if to_dest:
        out = torch_scatter.scatter_softmax(
            src = edge_feat, 
            index = edge_index[1], 
            dim = 0, 
            dim_size = num_nodes, 
        )
    else:
        out = torch_scatter.scatter_softmax(
            src = edge_feat, 
            index = edge_index[0], 
            dim = 0, 
            dim_size = num_nodes, 
        )
        
    assert out.shape == (num_edges, feat_dim) 
    
    return out 
