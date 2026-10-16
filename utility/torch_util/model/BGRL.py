from ..imports import * 

__all__ = [
    'BGRL', 
]


def _compute_cosine_similarity_loss(h1: FloatTensor, 
                                    h2: FloatTensor) -> FloatScalarTensor:
    N, D = h1.shape 
    assert h1.shape == h2.shape == (N, D) 
                                
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    loss = 1. - (h1 * h2).sum(dim=-1)
    loss = loss.mean()

    return loss


class BGRL(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 predictor: nn.Module):
        super().__init__()

        self.online_encoder = encoder

        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.reset_parameters()
        self.target_encoder.requires_grad_(False)

        self.predictor = predictor

    @torch.no_grad()
    def update_target_encoder(self, 
                              momentum: float):
        assert 0. <= momentum <= 1. 
        
        for src_param, target_param in itertools.zip_longest(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.mul_(momentum).add_(src_param.data, alpha=1.-momentum)

    def train_forward(self,
                      online_edge_index: EdgeIndex, 
                      online_node_feat: FloatTensor, 
                      target_edge_index: EdgeIndex, 
                      target_node_feat: FloatTensor) -> FloatScalarTensor:
        online_g = dgl.graph(tuple(online_edge_index), num_nodes=len(online_node_feat))
        online_g = dgl.add_self_loop(dgl.remove_self_loop(online_g))
        target_g = dgl.graph(tuple(target_edge_index), num_nodes=len(target_node_feat))
        target_g = dgl.add_self_loop(dgl.remove_self_loop(target_g))
        
        online_emb = self.online_encoder(dgl_g=online_g, node_feat=online_node_feat)

        online_pred = self.predictor(online_emb)

        with torch.no_grad():
            target_emb = self.target_encoder(dgl_g=target_g, node_feat=target_node_feat).detach()

        loss = _compute_cosine_similarity_loss(online_pred, target_emb) 

        return loss 
    
    @torch.no_grad() 
    def eval_forward(self,
                     edge_index: EdgeIndex, 
                     node_feat: FloatTensor) -> FloatTensor: 
        g = dgl.graph(tuple(edge_index), num_nodes=len(node_feat))
                     
        out = self.online_encoder(dgl_g=g, node_feat=node_feat).detach() 
        
        return out 
