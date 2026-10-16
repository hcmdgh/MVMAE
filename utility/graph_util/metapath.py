import torch 
from torch import Tensor
from torch_geometric.data import HeteroData

EdgeType = tuple[str, str, str]


def convert_node_type_list_list_to_metapath_list(
    graph: HeteroData,
    node_type_list_list: list[list[str]],
) -> list[list[EdgeType]]:
    edge_type_mapping = dict() 

    for edge_type in graph.edge_index_dict.keys():
        src_node_type, _, dst_node_type = edge_type

        assert (src_node_type, dst_node_type) not in edge_type_mapping
        edge_type_mapping[src_node_type, dst_node_type] = edge_type

    metapath_list = []

    for node_type_list in node_type_list_list:
        metapath = [] 

        if len(node_type_list) > 1:
            for i in range(len(node_type_list) - 1):
                src_node_type = node_type_list[i]
                dst_node_type = node_type_list[i + 1]

                edge_type = edge_type_mapping[src_node_type, dst_node_type]
                metapath.append(edge_type)
        elif len(node_type_list) == 1:
            node_type = node_type_list[0]
            metapath.append((node_type, 'self', node_type))
        else:
            raise ValueError

        metapath_list.append(metapath)

    return metapath_list


def generate_metapath_weighted_subgraphs(
    num_nodes_dict: dict[str, int],
    edge_index_dict: dict[EdgeType, Tensor],
    metapath_list: list[list[EdgeType]],
) -> tuple[list[Tensor], list[Tensor]]:
    device = next(iter(edge_index_dict.values())).device

    subgraph_edge_index_list = []
    subgraph_edge_weight_list = []

    for metapath in metapath_list:
        adj_mat_prod = None 

        for edge_type in metapath:
            src_node_type, _, dst_node_type = edge_type
            edge_index = edge_index_dict[edge_type]
            num_src_nodes = num_nodes_dict[src_node_type]
            num_dst_nodes = num_nodes_dict[dst_node_type]

            adj_mat = torch.sparse_coo_tensor(
                indices = edge_index,
                values = torch.ones(edge_index.shape[1], dtype=torch.float, device=device),
                size = (num_src_nodes, num_dst_nodes),
            )

            if adj_mat_prod is None:
                adj_mat_prod = adj_mat 
            else:
                adj_mat_prod = adj_mat_prod @ adj_mat

            adj_mat_prod = adj_mat_prod.coalesce()

        assert adj_mat_prod is not None
        subgraph_edge_index = adj_mat_prod.indices()
        subgraph_edge_weight = adj_mat_prod.values()
        num_edges = subgraph_edge_index.shape[1]
        assert subgraph_edge_index.shape == (2, num_edges)
        assert subgraph_edge_weight.shape == (num_edges,)

        subgraph_edge_index_list.append(subgraph_edge_index)
        subgraph_edge_weight_list.append(subgraph_edge_weight)

    return subgraph_edge_index_list, subgraph_edge_weight_list


def generate_metapath_subgraphs(
    num_nodes_dict: dict[str, int],
    edge_index_dict: dict[EdgeType, Tensor],
    metapath_list: list[list[EdgeType]],
) -> list[Tensor]:
    return generate_metapath_weighted_subgraphs(
        num_nodes_dict = num_nodes_dict,
        edge_index_dict = edge_index_dict,
        metapath_list = metapath_list,
    )[0]
