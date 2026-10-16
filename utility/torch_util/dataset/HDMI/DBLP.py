from ...common import * 
from ...util.device import get_device, to_device
from ...graph.self_loop import add_self_loop as add_self_loop
from ...math.scale import standardize_scale_feature


@dataclass 
class DBLP:
    device: torch.device 
    edge_index_dict: dict[EdgeType, EdgeIndex] 
    num_nodes_dict: dict[NodeType, int] 
    node_feat_dict: dict[NodeType, FloatTensor]
    node_feat_dim_dict: dict[NodeType, int]  
    infer_ntype: NodeType 
    node_label: IntTensor 
    node_train_mask: BoolTensor
    node_val_mask: BoolTensor
    node_test_mask: BoolTensor
    
    def __init__(self,
                 device: torch.device):
        with open(os.path.expanduser("~/dataset/HDMI/processed/DBLP_metapath.dict.pkl"), 'rb') as fp:
            graph_info = pickle.load(fp) 
            
        self.edge_index_dict = {
            ('paper', 'PP', 'paper'): graph_info['PP_edge_index'], 
            ('paper', 'PAP', 'paper'): graph_info['PAP_edge_index'], 
            ('paper', 'PTP', 'paper'): graph_info['PTP_edge_index'], 
            ('paper', 'PATAP', 'paper'): graph_info['PATAP_edge_index'], 
        }
        self.num_nodes_dict = { 'paper': len(graph_info['node_feat']) }
        self.node_feat_dict = { 'paper': graph_info['node_feat'] } 
        self.node_feat_dim_dict = { 'paper': graph_info['node_feat'].shape[-1] }
        self.infer_ntype = 'paper' 
        self.node_label = graph_info['node_label']
        self.node_train_mask = graph_info['node_train_mask']
        self.node_val_mask = graph_info['node_val_mask']
        self.node_test_mask = graph_info['node_test_mask']
        
        self.add_self_loop() 

        self.to(device) 

    def to(self,
           device: torch.device): 
        self.device = device 
        self.edge_index_dict = to_device(self.edge_index_dict, device=device)
        self.node_feat_dict = to_device(self.node_feat_dict, device=device)
        self.node_label = to_device(self.node_label, device=device)
        # self.node_train_mask = to_device(self.node_train_mask, device=device)
        # self.node_val_mask = to_device(self.node_val_mask, device=device)
        # self.node_test_mask = to_device(self.node_test_mask, device=device)
    
    def fill_random_feature(self):
        for ntype, num_nodes in self.num_nodes_dict.items(): 
            if ntype != self.infer_ntype:
                self.node_feat_dim_dict[ntype] = self.node_feat_dim_dict[self.infer_ntype]
                self.node_feat_dict[ntype] = torch.zeros(num_nodes, self.node_feat_dim_dict[ntype], device=self.device).uniform_(-0.5, 0.5)

    def normalize_node_feature(self): 
        for ntype, node_feat in self.node_feat_dict.items(): 
            node_feat = standardize_scale_feature(node_feat) 
            self.node_feat_dict[ntype] = node_feat 

    def add_self_loop(self): 
        for etype, edge_index in self.edge_index_dict.items(): 
            edge_index = add_self_loop(edge_index=edge_index, num_nodes=self.num_nodes_dict[self.infer_ntype])
            self.edge_index_dict[etype] = edge_index 
