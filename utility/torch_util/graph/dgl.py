import dgl 

from ..common import * 


def create_dgl_heterograph(edge_index_dict: dict[EdgeType, EdgeIndex],
                           num_nodes_dict: dict[NodeType, int],
                           node_attr_dict: Optional[dict[NodeType, dict[str, Tensor]]] = None) -> dgl.DGLHeteroGraph:
    hg = dgl.heterograph(
        { k: tuple(v) for k, v in edge_index_dict.items() }, 
        num_nodes_dict = num_nodes_dict, 
    )
    
    if node_attr_dict:
        for ntype in node_attr_dict: 
            for attr_name, attr_value in node_attr_dict[ntype].items():
                hg.nodes[ntype].data[attr_name] = attr_value
    
    return hg 
