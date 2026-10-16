from ..common import * 


def generate_message(node_feat: FloatTensor, 
                     edge_index: EdgeIndex,
                     from_dest: bool = False) -> FloatTensor:
    if node_feat.ndim == 2:
        num_nodes, feat_dim = node_feat.shape 
        num_heads = None 
    else: 
        num_nodes, num_heads, feat_dim = node_feat.shape 
        
    num_edges = edge_index.shape[-1] 
    assert edge_index.shape == (2, num_edges) 
                     
    if not from_dest: 
        msg = node_feat[edge_index[0]] 
    else:
        msg = node_feat[edge_index[1]] 

    assert msg.shape == (num_edges, feat_dim) or msg.shape == (num_edges, num_heads, feat_dim) 
        
    return msg 


def aggregate_message(message: FloatTensor, 
                      edge_index: EdgeIndex,
                      num_nodes: int, 
                      to_dest: bool = True,
                      reduce: str = 'mean') -> FloatTensor:
    if message.ndim == 2:
        num_edges, feat_dim = message.shape 
        num_heads = None 
    else:
        num_edges, num_heads, feat_dim = message.shape 
        
    assert edge_index.shape == (2, num_edges) 
    
    if to_dest:
        out = torch_scatter.scatter(
            src = message, 
            index = edge_index[1], 
            dim = 0, 
            dim_size = num_nodes, 
            reduce = reduce, 
        )
    else:
        out = torch_scatter.scatter(
            src = message, 
            index = edge_index[0], 
            dim = 0, 
            dim_size = num_nodes, 
            reduce = reduce, 
        )
        
    assert out.shape == (num_nodes, feat_dim) or out.shape == (num_nodes, num_heads, feat_dim)  
    
    return out 
