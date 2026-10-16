from ...common import * 
from ...util.device import get_device, to_device
from ...math.scale import standardize_scale_feature


@dataclass 
class OgbnMag:
    device: torch.device 
    edge_index_dict: dict[EdgeType, EdgeIndex] 
    num_nodes_dict: dict[NodeType, int] 
    node_feat_dict: dict[NodeType, FloatTensor]
    node_feat_dim_dict: dict[NodeType, int]  
    infer_ntype: NodeType 
    node_label: IntTensor 
    node_year: IntTensor 
    node_train_mask: BoolTensor
    node_val_mask: BoolTensor
    node_test_mask: BoolTensor
    
    def __init__(self,
                 device: torch.device):
        with open(os.path.expanduser("~/dataset/OGB/ogbn-mag/processed/hg_undirected.dict.pkl"), 'rb') as fp:
            graph_info = pickle.load(fp) 
            
        self.edge_index_dict = graph_info['edge_index_dict']
        self.num_nodes_dict = graph_info['num_nodes_dict']
        self.node_feat_dict = { 'paper': graph_info['paper_feat'] } 
        self.node_feat_dim_dict = { 'paper': self.node_feat_dict['paper'].shape[-1] }
        self.infer_ntype = 'paper' 
        self.node_label = graph_info['paper_label']
        self.node_year = graph_info['paper_year'] 
        self.node_train_mask = graph_info['paper_train_mask']
        self.node_val_mask = graph_info['paper_val_mask']
        self.node_test_mask = graph_info['paper_test_mask']

        self.to(device) 

    def to(self,
           device: torch.device): 
        self.device = device 
        self.edge_index_dict = to_device(self.edge_index_dict, device=device)
        self.node_feat_dict = to_device(self.node_feat_dict, device=device)
        self.node_label = to_device(self.node_label, device=device)
        self.node_year = to_device(self.node_year, device=device)
        # self.node_train_mask = to_device(self.node_train_mask, device=device)
        # self.node_val_mask = to_device(self.node_val_mask, device=device)
        # self.node_test_mask = to_device(self.node_test_mask, device=device)
    
    def fill_random_feature(self):
        for ntype, num_nodes in self.num_nodes_dict.items(): 
            if ntype != self.infer_ntype:
                self.node_feat_dim_dict[ntype] = self.node_feat_dim_dict[self.infer_ntype]
                self.node_feat_dict[ntype] = torch.zeros(num_nodes, self.node_feat_dim_dict[ntype], device=self.device).uniform_(-0.5, 0.5)

    def fill_propagated_feature(self): 
        with open(os.path.expanduser("~/dataset/OGB/ogbn-mag/processed/propagated_node_feature.dict.pkl"), 'rb') as fp:
            other_node_feat_dict = pickle.load(fp) 
            
        self.node_feat_dict['author'] = other_node_feat_dict['author_feat'].to(self.device) 
        self.node_feat_dict['field'] = other_node_feat_dict['field_feat'].to(self.device) 
        self.node_feat_dict['institution'] = other_node_feat_dict['institution_feat'].to(self.device) 

    def normalize_node_feature(self): 
        for ntype, node_feat in self.node_feat_dict.items(): 
            node_feat = standardize_scale_feature(node_feat) 
            self.node_feat_dict[ntype] = node_feat 


@dataclass 
class OgbnMagPaper:
    device: torch.device 
    edge_index: EdgeIndex
    node_feat: FloatTensor
    node_label: IntTensor 
    node_year: IntTensor 
    node_train_mask: BoolTensor
    node_val_mask: BoolTensor
    node_test_mask: BoolTensor
    
    def __init__(self,
                 device: Any):
        with open(os.path.expanduser("~/dataset/OGB/ogbn-mag/processed/g_paper.dict.pkl"), 'rb') as fp:
            graph_info = pickle.load(fp) 
            
        self.edge_index = graph_info['edge_index']
        self.node_feat = graph_info['node_feat']
        self.node_label = graph_info['node_label']
        self.node_year = graph_info['node_year'] 
        self.node_train_mask = graph_info['node_train_mask']
        self.node_val_mask = graph_info['node_val_mask']
        self.node_test_mask = graph_info['node_test_mask']

        self.to(device)
        
    def num_nodes(self) -> int:
        return len(self.node_feat) 
    
    def num_edges(self) -> int:
        return self.edge_index.shape[-1]  

    def to(self,
           device: Any): 
        self.device = torch.device(device) 
        
        self.edge_index = self.edge_index.to(self.device) 
        self.node_feat = self.node_feat.to(device)
        self.node_label = self.node_label.to(device)
        self.node_year = self.node_year.to(device)
        self.node_train_mask = self.node_train_mask.to(device)
        self.node_val_mask = self.node_val_mask.to(device)
        self.node_test_mask = self.node_test_mask.to(device)
    
    def to_undirected(self):
        pass  # do nothing 
