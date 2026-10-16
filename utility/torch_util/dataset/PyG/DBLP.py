from ...common import * 
from ...util.device import to_device


@dataclass 
class DBLP:
    device: torch.device 
    node_feat_dict: dict[NodeType, FloatTensor]
    edge_index_dict: dict[EdgeType, EdgeIndex]
    infer_ntype: NodeType  
    node_label: IntTensor 
    node_train_mask: BoolTensor
    node_val_mask: BoolTensor
    node_test_mask: BoolTensor
    
    def __init__(self,
                 device: Any):
        with open(os.path.expanduser("~/dataset/PyG/DBLP/processed/hg.dict.pkl"), 'rb') as fp:
            graph_info = pickle.load(fp) 
            
        self.node_feat_dict = graph_info['node_feat_dict']
        self.edge_index_dict = graph_info['edge_index_dict']
        self.infer_ntype = 'author'
        self.node_label = graph_info['author_label']
        self.node_train_mask = graph_info['author_train_mask']
        self.node_val_mask = graph_info['author_val_mask']
        self.node_test_mask = graph_info['author_test_mask']

        self.to(device) 
        
    def num_nodes_dict(self) -> dict[NodeType, int]:
        return { ntype: len(self.node_feat_dict[ntype]) for ntype in self.node_feat_dict }

    def to(self,
           device: Any): 
        self.device = torch.device(device)
        
        self.node_feat_dict = to_device(self.node_feat_dict, device=self.device)
        self.edge_index_dict = to_device(self.edge_index_dict, device=self.device)
        self.node_label = to_device(self.node_label, device=self.device)
        self.node_train_mask = to_device(self.node_train_mask, device=self.device)
        self.node_val_mask = to_device(self.node_val_mask, device=self.device)
        self.node_test_mask = to_device(self.node_test_mask, device=self.device)
    