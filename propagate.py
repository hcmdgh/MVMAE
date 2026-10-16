import torch 
from torch import Tensor 

from sparse_util import SparseTensor

EdgeType = tuple[str, str, str]


def propagate_along_metapaths(
    edge_index_dict: dict[EdgeType, Tensor],
    node_feat_dict: dict[str, Tensor],
    metapath_list: list[list[EdgeType]],
    normalize_adj_mat: str, 
    on_cpu: bool = True,
) -> list[Tensor]:
    device = next(iter(edge_index_dict.values())).device 

    if on_cpu:
        edge_index_dict = {
            edge_type: edge_index.cpu() 
            for edge_type, edge_index in edge_index_dict.items()
        }

        node_feat_dict = {
            node_type: node_feat.cpu() 
            for node_type, node_feat in node_feat_dict.items()
        }

    num_nodes_dict = {
        node_type: len(node_feat) 
        for node_type, node_feat in node_feat_dict.items()
    }

    adj_mat_t_dict = {
        edge_type: SparseTensor.from_edge_index(
            edge_index = edge_index,
            num_nodes = (num_nodes_dict[edge_type[0]], num_nodes_dict[edge_type[-1]]),
        ).t() 
        for edge_type, edge_index in edge_index_dict.items()
    }

    if normalize_adj_mat == 'none':
        pass 
    elif normalize_adj_mat == 'sage':
        for edge_type, adj_mat_t in adj_mat_t_dict.items():
            adj_mat_t_dict[edge_type] = adj_mat_t.row_normalize()
    elif normalize_adj_mat == 'gcn':
        for edge_type, adj_mat_t in adj_mat_t_dict.items():
            adj_mat_t_dict[edge_type] = adj_mat_t.gcn_normalize()
    else:
        raise ValueError 

    output_list = [] 
    
    for metapath in metapath_list:
        if len(metapath) == 1 and metapath[0][1] == 'self':
            node_type = metapath[0][0]
            output = node_feat_dict[node_type]
            output_list.append(output)
        else:
            start_node_type = metapath[0][0]
            propagated_node_feat = node_feat_dict[start_node_type]
            
            for edge_type in metapath:
                adj_mat_t = adj_mat_t_dict[edge_type]

                propagated_node_feat = adj_mat_t.matmul(propagated_node_feat)
                
            output_list.append(propagated_node_feat)

    assert len(output_list) == len(metapath_list)

    output_list = [
        output.to(device) 
        for output in output_list
    ]
        
    return output_list 
