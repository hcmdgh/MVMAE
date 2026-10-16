import os 
import pickle 

from graph_util.metapath import convert_node_type_list_list_to_metapath_list
from .base import HeteroGraphDataset


class RCDDSubgraphDataset(HeteroGraphDataset):
    def __init__(
        self,
        metapath_schema: str = '',
    ):
        with open(os.path.expanduser(f'~/dataset/graph/hetero_graph/RCDD/processed_data/subgraph_IBI_IFI.pkl'), 'rb') as r:
            graph = pickle.load(r)

        graph['item'].val_mask = graph['item'].test_mask

        metric_name_list = ['ap', 'micro_f1', 'macro_f1']

        hop_0_metapath_list = [
            ['item'],
        ]

        hop_1_metapath_list = [
            ['b', 'item'],
            ['f', 'item'],
        ]
        
        hop_2_metapath_list = [
            ['item', 'b', 'item'],
            ['item', 'f', 'item'],
        ]

        hop_4_metapath_list = [
            ['item', 'b', 'item', 'b', 'item'],
            ['item', 'f', 'item', 'f', 'item'],
        ]

        if not metapath_schema:
            metapath_list = []
        elif metapath_schema == '2_hop':
            metapath_list = hop_2_metapath_list
        elif metapath_schema == '2_4_hop':
            metapath_list = hop_2_metapath_list + hop_4_metapath_list
        elif metapath_schema == '0_2_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list
        elif metapath_schema == '0_2_4_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list + hop_4_metapath_list
        elif metapath_schema == '0_1_2_hop':
            metapath_list = hop_0_metapath_list + hop_1_metapath_list + hop_2_metapath_list
        elif metapath_schema == '0_1_2_4_hop':
            metapath_list = hop_0_metapath_list + hop_1_metapath_list + hop_2_metapath_list + hop_4_metapath_list
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
        )


class RCDDDataset(HeteroGraphDataset):
    def __init__(
        self,
        metapath_schema: str = '',
    ):
        with open(os.path.expanduser(f'~/dataset/graph/hetero_graph/RCDD/processed_data/subgraph_IBI_IFI_v2.pkl'), 'rb') as r:
            graph = pickle.load(r)

        graph['item'].val_mask = graph['item'].test_mask

        metric_name_list = ['ap', 'micro_f1', 'macro_f1']

        hop_0_metapath_list = [
            ['item'],
        ]

        hop_1_metapath_list = [
            ['b', 'item'],
            ['f', 'item'],
        ]
        
        hop_2_metapath_list = [
            ['item', 'b', 'item'],
            ['item', 'f', 'item'],
        ]

        hop_4_metapath_list = [
            ['item', 'b', 'item', 'b', 'item'],
            ['item', 'f', 'item', 'f', 'item'],
        ]

        if not metapath_schema:
            metapath_list = []
        elif metapath_schema == '2_hop':
            metapath_list = hop_2_metapath_list
        elif metapath_schema == '2_4_hop':
            metapath_list = hop_2_metapath_list + hop_4_metapath_list
        elif metapath_schema == '0_2_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list
        elif metapath_schema == '0_2_4_hop':
            metapath_list = hop_0_metapath_list + hop_2_metapath_list + hop_4_metapath_list
        elif metapath_schema == '0_1_2_hop':
            metapath_list = hop_0_metapath_list + hop_1_metapath_list + hop_2_metapath_list
        elif metapath_schema == '0_1_2_4_hop':
            metapath_list = hop_0_metapath_list + hop_1_metapath_list + hop_2_metapath_list + hop_4_metapath_list
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
        )
