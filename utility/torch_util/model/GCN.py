from ..imports import * 
from ..graph import generate_message, aggregate_message, in_degree 
from ..graph import add_self_loop as _add_self_loop   
# from ..dataloader import NeighborLoader

__all__ = [
    'GCN', 
    'GCNConv', 
    'compute_gcn_norm', 
]


def compute_gcn_norm(edge_index: EdgeIndex, 
                     num_nodes: int) -> FloatTensor:
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    
    deg = in_degree(edge_index=edge_index, num_nodes=num_nodes).float() 
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)

    edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]] 
    assert edge_weight.shape == (num_edges,) 

    return edge_weight 


class GCNConv(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int):
        super().__init__()

        self.in_dim = in_dim 
        self.out_dim = out_dim 
        
        self.W_fc = nn.Linear(in_dim, out_dim, bias=False) 

        self.bias = Parameter(torch.zeros(out_dim)) 
        
    def forward(self, 
                node_feat: FloatTensor, 
                edge_index: EdgeIndex,
                edge_weight: FloatTensor) -> FloatTensor: 
        assert node_feat.shape == (num_nodes := len(node_feat), self.in_dim) 
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
        assert edge_weight.shape == (num_edges,) 
        
        h = self.W_fc(node_feat)
        
        msg = generate_message(
            node_feat = h, 
            edge_index = edge_index,
        )
        
        msg.mul_(edge_weight.view(-1, 1)) 
        
        out = aggregate_message(
            message = msg, 
            edge_index = edge_index, 
            num_nodes = num_nodes, 
            reduce = 'sum', 
        )
        
        out.add_(self.bias) 
            
        return out 
        

class GCN(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 add_self_loop: bool = True, 
                 batch_norm: bool = False, 
                 activation: nn.Module = nn.ReLU(), 
                 dropout: float = 0.):
        super().__init__()

        self.num_layers = num_layers 
        self.add_self_loop = add_self_loop 
        
        self.conv_list = nn.ModuleList([
            GCNConv(
                in_dim = in_dim, 
                out_dim = hidden_dim,
            ), 
            *[
                GCNConv(
                    in_dim = hidden_dim, 
                    out_dim = hidden_dim,
                ) 
                for _ in range(num_layers - 2) 
            ],
            GCNConv(
                in_dim = hidden_dim, 
                out_dim = out_dim,
            ), 
        ])
        assert len(self.conv_list) == num_layers >= 2 
        
        if batch_norm: 
            self.bn_list = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) 
                for _ in range(num_layers - 1) 
            ])
        else:
            self.bn_list = None 
            
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 

    def forward(self,
                node_feat: FloatTensor, 
                edge_index: EdgeIndex) -> FloatTensor: 
        num_nodes, feat_dim = node_feat.shape  
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
        
        if self.add_self_loop: 
            edge_index = _add_self_loop(edge_index=edge_index, num_nodes=num_nodes) 
            
        edge_weight = compute_gcn_norm(edge_index=edge_index, num_nodes=num_nodes) 
                
        h = node_feat
        
        for l in range(self.num_layers): 
            h = self.conv_list[l](node_feat=h, edge_index=edge_index, edge_weight=edge_weight)         
            
            if l < self.num_layers - 1: 
                if self.bn_list is not None: 
                    h = self.bn_list[l](h) 
                h = self.activation(h) 
                h = self.dropout(h)
                            
        return h 

    @torch.no_grad() 
    def inference(self,
                  node_feat: FloatTensor,
                  edge_index: EdgeIndex, 
                  batch_size: int) -> FloatTensor: 
        device = node_feat.device 
        num_nodes, feat_dim = node_feat.shape 
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
        
        inference_loader = NeighborLoader(
            edge_index = edge_index, 
            num_nodes = num_nodes, 
            target_node_idx = None,
            num_neighbors = [-1], 
            batch_size = batch_size, 
            shuffle = False,  
        )

        with tqdm(desc='Inference', total=num_nodes * self.num_layers) as bar:
            h_all = node_feat 
            
            for l in range(self.num_layers):
                h_list = []

                for batch in inference_loader:
                    h_batch = h_all[batch.node_idx] 
                    edge_index_batch = batch.edge_index.to(device) 
                    
                    h_batch = self.conv_list[l](src_feat=h_batch, dest_feat=h_batch, edge_index=edge_index_batch)[batch.target_node_mask] 

                    if l < self.num_layers - 1: 
                        if self.bn_list is not None: 
                            h_batch = self.bn_list[l](h_batch) 
                        h_batch = self.activation(h_batch) 
                        h_batch = self.dropout(h_batch)

                    h_list.append(h_batch) 

                    bar.update(batch.num_target_nodes) 

                h_all = torch.cat(h_list, dim=0)

        return h_all
