from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData 

from ...imports import * 
from ...util import get_device 

__all__ = [
    'HeteroGraphSimpleNeighborLoader_PyG',
]


class HeteroGraphSimpleNeighborLoader_PyG(NeighborLoader):
    def __init__(self, *, 
                 edge_index_dict: dict[EdgeType, EdgeIndex], 
                 node_feat_dict: dict[NodeType, Tensor], 
                 node_label: Tensor, 
                 predict_ntype: NodeType,  
                 predict_nodes: Union[IntTensor, BoolTensor, None], 
                 batch_size: int,
                 num_neighbors: list[int],
                 shuffle: bool,
                 device: Any = None, 
                 num_workers: int = 0,
                 pin_memory: Optional[bool] = None): 
        if device is not None: 
            self.device = torch.device(device) 
        else:
            self.device = get_device() 
                 
        hetero_data = HeteroData() 
        
        for etype, edge_index in edge_index_dict.items(): 
            hetero_data[etype].edge_index = edge_index.to(self.device).contiguous() 
            
        for ntype, node_feat in node_feat_dict.items(): 
            hetero_data[ntype].feat = node_feat.to(self.device)
            
        hetero_data[predict_ntype].label = node_label.to(self.device)
            
        super().__init__(
            data = hetero_data, 
            num_neighbors = num_neighbors, 
            input_nodes = (predict_ntype, predict_nodes), 
            batch_size = batch_size, 
            shuffle = shuffle, 
            num_workers = num_workers, 
            pin_memory = pin_memory, 
        )

    def __iter__(self) -> Iterable[tuple[dict[EdgeType, EdgeIndex], dict[NodeType, Tensor], Tensor, BoolTensor]]: 
        for batch in super().__iter__(): 
            batch = batch.to_dict() 

            node_feat_dict_batch = dict() 
            edge_index_dict_batch = dict() 
            node_label_batch = None 
            target_mask = None 

            for key in batch: 
                if isinstance(key, str): 
                    node_feat_dict_batch[key] = batch[key]['feat'] 
                    
                    if 'batch_size' in batch[key]: 
                        assert node_label_batch is None 
                        node_label_batch = batch[key]['label']
                        
                        target_size = batch[key]['batch_size']
                        src_size = len(node_label_batch)
                        target_mask = torch.zeros(src_size, dtype=torch.bool, device=node_label_batch.device) 
                        target_mask[:target_size] = True 
                elif isinstance(key, tuple): 
                    edge_index_dict_batch[key] = batch[key]['edge_index'] 
                else:
                    raise AssertionError 
                
            assert node_label_batch is not None and target_mask is not None 
            
            yield edge_index_dict_batch, node_feat_dict_batch, node_label_batch, target_mask 
