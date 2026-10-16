import torch 
from torch import Tensor
from torch_geometric.data import HeteroData


def build_onehot_feat(
    num_nodes: int,
) -> Tensor:
    return torch.sparse_coo_tensor(
        indices = torch.arange(num_nodes).reshape(1, num_nodes).repeat(2, 1),
        values = torch.ones(num_nodes, dtype=torch.float),
        size = (num_nodes, num_nodes),
    )


def init_absent_node_feat(
    graph: HeteroData,
    mode: str,
) -> HeteroData:
    if mode == 'none':
        return graph

    for node_type, num_nodes in graph.num_nodes_dict.items():
        if 'x' not in graph[node_type]:
            if mode == 'onehot':
                x = build_onehot_feat(num_nodes=num_nodes)
            elif mode.startswith('randn_'):
                feat_dim = int(mode.removeprefix('randn_'))
                x = torch.randn(num_nodes, feat_dim)
            else:
                raise ValueError 

            graph[node_type].x = x 

    return graph 
