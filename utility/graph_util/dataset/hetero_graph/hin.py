import os 
import pickle 

from graph_util.metapath import convert_node_type_list_list_to_metapath_list
from .base import HeteroGraphDataset


class HINAMinerDataset(HeteroGraphDataset):
    def __init__(
        self,
        split: int,
        metapath_schema: str = '',
        node_feat_init: str = 'onehot',
    ):
        with open(os.path.expanduser(f'~/dataset/graph/hetero_graph/HIN_AMiner/processed_data/graph_with_split.pkl'), 'rb') as r:
            graph = pickle.load(r)

        graph.train_mask = graph.pop('train_mask_dict')[split]
        graph.val_mask = graph.pop('val_mask_dict')[split]
        graph.test_mask = graph.pop('test_mask_dict')[split]

        metric_name_list = ['micro_f1', 'macro_f1']

        hop_0_metapath_list = [
            ['paper'],
        ]

        hop_2_metapath_list = [
            ['paper', 'author', 'paper'],
            ['paper', 'conference', 'paper'],
            ['paper', 'reference', 'paper'],
        ]

        hop_4_metapath_list = [
            ['paper', 'author', 'paper', 'author', 'paper'],
            ['paper', 'conference', 'paper', 'conference', 'paper'],
            ['paper', 'reference', 'paper', 'reference', 'paper'],
        ]

        if not metapath_schema:
            metapath_list = []
        elif metapath_schema == '2_hop':
            metapath_list = hop_2_metapath_list
        elif metapath_schema == '0_2_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list
        elif metapath_schema == '2_4_hop':
            metapath_list = hop_2_metapath_list + hop_4_metapath_list
        elif metapath_schema == '0_2_4_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list + hop_4_metapath_list
        else:
            raise ValueError
        
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
