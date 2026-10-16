from torch_geometric.nn import HGTConv 

from ..common import * 


class HGT_PyG(nn.Module):
    def __init__(self, 
                 in_dim_dict: dict[NodeType, int], 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_heads: int, 
                 num_layers: int,
                 ntypes: set[NodeType],
                 etypes: set[EdgeType]):
        super().__init__()
        
        self.ntypes = ntypes 
        self.etypes = etypes 
        
        self.in_lin_dict = nn.ModuleDict({
            ntype: nn.Linear(in_dim_dict[ntype], hidden_dim) 
            for ntype in ntypes 
        }) 
        
        self.conv_list = nn.ModuleList([
            HGTConv(
                in_channels = hidden_dim, 
                out_channels = hidden_dim, 
                metadata = (list(ntypes), list(etypes)), 
                heads = num_heads, 
            )
            for _ in range(num_layers) 
        ])
        
        self.out_lin_dict = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim) 
            for ntype in ntypes 
        }) 

    def forward(self, 
                node_feat_dict: dict[NodeType, FloatTensor], 
                edge_index_dict: dict[EdgeType, EdgeIndex]) -> dict[NodeType, FloatTensor]:
        h_dict = {
            ntype: torch.relu( self.in_lin_dict[ntype](node_feat) ) 
            for ntype, node_feat in node_feat_dict.items()
        }

        for conv in self.conv_list:
            h_dict = conv(x_dict=h_dict, edge_index_dict=edge_index_dict) 
            
        out_dict = {
            ntype: self.out_lin_dict[ntype](h)
            for ntype, h in h_dict.items()
        }

        return out_dict 
