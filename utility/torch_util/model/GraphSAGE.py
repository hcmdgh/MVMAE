from ..imports import * 
from ..graph import generate_message, aggregate_message 
# from ..dataloader import NeighborLoader

__all__ = [
    'GraphSAGE', 
    'GraphSAGEConv', 
]


class GraphSAGEConv(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 project: bool = False,
                 normalize: bool = False):
        super().__init__()

        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.normalize = normalize 
        
        if project: 
            self.project_fc = nn.Linear(in_dim, in_dim) 
        else:
            self.project_fc = None 
            
        self.out_fc = nn.Linear(in_dim, out_dim)

        self.root_fc = nn.Linear(in_dim, out_dim) 
        
    def forward(self, 
                src_feat: FloatTensor, 
                dest_feat: FloatTensor, 
                edge_index: EdgeIndex) -> FloatTensor: 
        assert src_feat.shape == (num_src_nodes := src_feat.shape[0], self.in_dim)
        assert dest_feat.shape == (num_dest_nodes := dest_feat.shape[0], self.in_dim)
        assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
        
        if self.project_fc: 
            src_feat = self.project_fc(src_feat) 
            src_feat = torch.relu(src_feat) 
            
        msg = generate_message(
            node_feat = src_feat, 
            edge_index = edge_index, 
        )
        assert msg.shape == (num_edges, self.in_dim) 
        
        out = aggregate_message(
            message = msg, 
            edge_index = edge_index, 
            num_nodes = num_dest_nodes, 
            reduce = 'mean', 
        )
        assert out.shape == (num_dest_nodes, self.in_dim)
        
        out = self.out_fc(out) 
        assert out.shape == (num_dest_nodes, self.out_dim)
        
        out = out + self.root_fc(dest_feat) 
        assert out.shape == (num_dest_nodes, self.out_dim)
        
        if self.normalize: 
            out = F.normalize(out, p=2., dim=-1) 
            
        return out 
        

class GraphSAGE(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 project: bool = False,
                 normalize: bool = False, 
                 batch_norm: bool = False, 
                 activation: nn.Module = nn.ReLU(), 
                 dropout: float = 0.):
        super().__init__()

        assert num_layers >= 2 
        self.num_layers = num_layers 
        
        self.conv_list = nn.ModuleList([
            GraphSAGEConv(
                in_dim = in_dim, 
                out_dim = hidden_dim,
                project = project, 
                normalize = normalize, 
            ), 
            *[
                GraphSAGEConv(
                    in_dim = hidden_dim, 
                    out_dim = hidden_dim,
                    project = project, 
                    normalize = normalize, 
                ) 
                for _ in range(num_layers - 2) 
            ],
            GraphSAGEConv(
                in_dim = hidden_dim, 
                out_dim = out_dim,
                project = project, 
                normalize = normalize, 
            ), 
        ])
        
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
        h = node_feat
        
        for l in range(self.num_layers): 
            h = self.conv_list[l](src_feat=h, dest_feat=h, edge_index=edge_index)        
            
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
