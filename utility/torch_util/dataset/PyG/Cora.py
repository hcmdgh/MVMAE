from ...common import * 
from ...util.device import get_device, to_device
from ...graph.undirected import to_undirected 
from ...graph.self_loop import add_self_loop 


@dataclass 
class Cora:
    device: torch.device 
    edge_index: EdgeIndex 
    node_feat: FloatTensor 
    node_label: IntTensor 
    node_train_mask: BoolTensor
    node_val_mask: BoolTensor
    node_test_mask: BoolTensor
    
    def __init__(self,
                 device: torch.device):
        with open(os.path.expanduser("~/dataset/PyG/Cora/processed/g.dict.pkl"), 'rb') as fp:
            graph_info = pickle.load(fp) 
            
        self.edge_index = graph_info['edge_index']
        self.node_feat = graph_info['node_feat'] 
        self.node_label = graph_info['node_label']
        self.node_train_mask = graph_info['node_train_mask']
        self.node_val_mask = graph_info['node_val_mask']
        self.node_test_mask = graph_info['node_test_mask']

        self.to(device) 

    def to(self,
           device: torch.device): 
        self.device = device 
        self.edge_index = self.edge_index.to(self.device) 
        self.node_feat = self.node_feat.to(self.device) 
        self.node_label = self.node_label.to(self.device) 
        self.node_train_mask = self.node_train_mask.to(self.device)
        self.node_val_mask = self.node_val_mask.to(self.device)
        self.node_test_mask = self.node_test_mask.to(self.device)
    
    def to_undirected(self): 
        self.edge_index = to_undirected(self.edge_index)
        
    def add_self_loop(self): 
        self.edge_index = add_self_loop(edge_index=self.edge_index, num_nodes=len(self.node_feat)) 
