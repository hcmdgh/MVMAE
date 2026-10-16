import torch 
from torch import Tensor
from torch_geometric.data import HeteroData
from dataclasses import dataclass
from typing import Callable

from graph_util.metric import compute_micro_f1, compute_macro_f1, compute_ap
from graph_util.init import init_absent_node_feat 
from graph_util.normalize import normalize_feature

EdgeType = tuple[str, str, str]


@dataclass
class HeteroGraphDataset:
    graph: HeteroData
    metapath_list: list[list[EdgeType]]
    metric_name_list: list[str]
    node_feat_init: str = 'none'

    def __post_init__(self):
        self.graph = init_absent_node_feat(
            graph = self.graph,
            mode = self.node_feat_init,
        )

        print(f"graph = {self.graph}\n")

    def num_nodes(self) -> int:
        return int(self.graph.num_nodes)
    
    def num_edges(self) -> int:
        return int(self.graph.num_edges)
    
    def num_nodes_dict(self) -> dict:
        return self.graph.num_nodes_dict
    
    def edge_index_dict(self) -> dict:
        return self.graph.edge_index_dict
    
    def node_type_set(self) -> set[str]:
        return set(self.num_nodes_dict().keys())
    
    def edge_type_set(self) -> set[tuple[str, str, str]]:
        return set(self.edge_index_dict().keys())
    
    def target_node_type(self) -> str:
        y_dict = self.graph.y_dict
        assert len(y_dict) == 1

        return list(y_dict.keys())[0]

    def target_num_nodes(self) -> int:
        target_node_type = self.target_node_type()
        num_nodes_dict = self.num_nodes_dict()

        return num_nodes_dict[target_node_type]
    
    def target_node_label(self) -> Tensor:
        y_dict = self.graph.y_dict
        assert len(y_dict) == 1

        return list(y_dict.values())[0]
    
    def target_node_feat(self) -> Tensor:
        target_node_type = self.target_node_type()
        node_feat_dict = self.node_feat_dict()
        target_node_feat = node_feat_dict[target_node_type]

        return target_node_feat
    
    def target_node_feat_dim(self) -> int:
        return self.target_node_feat().shape[-1]
    
    def num_classes(self) -> int:
        return int(self.target_node_label().max()) + 1
    
    def node_feat_dict(self) -> dict[str, Tensor]:
        return self.graph.x_dict
    
    def node_feat_dim_dict(self) -> dict[str, int]:
        return {
            node_type: node_feat.shape[-1]
            for node_type, node_feat in self.node_feat_dict().items()
        }
    
    def train_mask(self) -> Tensor:
        try:
            return self.graph.train_mask
        except (AttributeError, KeyError):
            return self.graph.train_mask_dict[self.target_node_type()]
        
    def val_mask(self) -> Tensor:
        try:
            return self.graph.val_mask
        except (AttributeError, KeyError):
            return self.graph.val_mask_dict[self.target_node_type()]
        
    def test_mask(self) -> Tensor:
        try:
            return self.graph.test_mask
        except (AttributeError, KeyError):
            return self.graph.test_mask_dict[self.target_node_type()]

    def main_metric_name(self) -> str:
        assert self.metric_name_list
        return self.metric_name_list[0]

    def metric_func_dict(self) -> dict[str, Callable]:
        metric_func_dict = dict() 

        for metric_name in self.metric_name_list:
            if metric_name == 'micro_f1':
                metric_func = compute_micro_f1
            elif metric_name == 'macro_f1':
                metric_func = compute_macro_f1
            elif metric_name == 'ap':
                metric_func = compute_ap 
            else:
                raise ValueError

            metric_func_dict[metric_name] = metric_func

        return metric_func_dict

    def cuda(self):
        self.graph = self.graph.cuda()

        return self 

    def normalize_node_feat_(self):
        for x in self.graph.x_dict.values():
            x.copy_( normalize_feature(x) )

        return self 
