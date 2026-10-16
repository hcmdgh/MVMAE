import os 
import pickle 

from graph_util.metapath import convert_node_type_list_list_to_metapath_list
from graph_util.init import init_absent_node_feat
from .base import HeteroGraphDataset


class HeCoDataset(HeteroGraphDataset):
    def __init__(
        self,
        name: str,
        split: int,
        metapath_list: list[list[str]],
        node_feat_init: str = 'none',
    ):
        with open(os.path.expanduser(f'~/dataset/graph/hetero_graph/HeCo_{name}/processed_data/graph.pkl'), 'rb') as r:
            graph = pickle.load(r)

        graph.train_mask = graph.pop('train_mask_dict')[split]
        graph.val_mask = graph.pop('val_mask_dict')[split]
        graph.test_mask = graph.pop('test_mask_dict')[split]

        metric_name_list = ['micro_f1', 'macro_f1']
        
        canonical_metapath_list = convert_node_type_list_list_to_metapath_list(
            graph = graph,
            node_type_list_list = metapath_list,
        )

        super().__init__(
            graph = graph,
            metapath_list = canonical_metapath_list,
            metric_name_list = metric_name_list,
            node_feat_init = node_feat_init,
        )


class HeCoACMDataset(HeCoDataset):
    def __init__(
        self,
        split: int,
        metapath_schema: str = '',
    ):
        if not metapath_schema:
            metapath_list = []
        elif metapath_schema == '2_hop':
            metapath_list = [
                ['paper', 'author', 'paper'],
                ['paper', 'subject', 'paper'],
            ]
        elif metapath_schema == '2_4_hop':
            metapath_list = [
                ['paper', 'author', 'paper'],
                ['paper', 'subject', 'paper'],
                ['paper', 'author', 'paper', 'author', 'paper'],
                ['paper', 'subject', 'paper', 'subject', 'paper'],
            ]
        else:
            raise ValueError
        
        super().__init__(
            name = 'ACM',
            split = split,
            metapath_list = metapath_list,
        )


class HeCoDBLPDataset(HeCoDataset):
    def __init__(
        self,
        split: int,
        metapath_schema: str = '',
    ):
        hop_0_metapath_list = [
            ['author'],
        ]

        hop_2_metapath_list = [
            ['author', 'paper', 'author'],
        ]

        hop_4_metapath_list = [
            ['author', 'paper', 'author', 'paper', 'author'],
            ['author', 'paper', 'conference', 'paper', 'author'],
            ['author', 'paper', 'term', 'paper', 'author'],
        ]

        hop_6_metapath_list = [
            ['author', 'paper', 'author', 'paper', 'author', 'paper', 'author'],
            ['author', 'paper', 'conference', 'paper', 'conference', 'paper', 'author'],
            ['author', 'paper', 'term', 'paper', 'term', 'paper', 'author'],
        ]

        if not metapath_schema:
            metapath_list = []
        elif metapath_schema == '2_4_hop':
            metapath_list = hop_2_metapath_list + hop_4_metapath_list
        elif metapath_schema == '0_2_4_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list + hop_4_metapath_list
        elif metapath_schema == '2_4_6_hop':
            metapath_list = hop_2_metapath_list + hop_4_metapath_list + hop_6_metapath_list
        elif metapath_schema == '0_2_4_6_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list + hop_4_metapath_list + hop_6_metapath_list
        else:
            raise ValueError
        
        super().__init__(
            name = 'DBLP',
            split = split,
            metapath_list = metapath_list,
        )


class HeCoFreebaseDataset(HeCoDataset):
    def __init__(
        self,
        split: int,
        metapath_schema: str = '',
        node_feat_init: str = 'onehot',
    ):
        hop_0_metapath_list = [
            ['movie'],
        ]

        hop_2_metapath_list = [
            ['movie', 'actor', 'movie'],
            ['movie', 'director', 'movie'],
            ['movie', 'writer', 'movie'],
        ]

        hop_4_metapath_list = [
            ['movie', 'actor', 'movie'],
            ['movie', 'director', 'movie'],
            ['movie', 'writer', 'movie'],
            ['movie', 'actor', 'movie', 'actor', 'movie'],
            ['movie', 'director', 'movie', 'director', 'movie'],
            ['movie', 'writer', 'movie', 'writer', 'movie'],
        ]

        if not metapath_schema:
            metapath_list = []
        elif metapath_schema == '2_hop':
            metapath_list = hop_2_metapath_list
        elif metapath_schema == '2_4_hop':
            metapath_list = hop_4_metapath_list
        elif metapath_schema == '0_2_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list
        elif metapath_schema == '0_2_4_hop':
            metapath_list = hop_0_metapath_list + hop_4_metapath_list
        else:
            raise ValueError
        
        super().__init__(
            name = 'Freebase',
            split = split,
            metapath_list = metapath_list,
            node_feat_init = node_feat_init,
        )
