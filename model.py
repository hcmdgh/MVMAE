import torch 
import torch.nn as nn
from torch import Tensor

from .encoder import Encoder
from .decoder import Decoder
from .mask import create_random_mask
from .loss import compute_reconstruction_loss


class MVMAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embedding_dim: int,
        sequence_len: int,
        mask_ratio: float,
        mask_token_learnable: bool,
        encoder_type: str,
        encoder_num_layers: int,
        encoder_num_heads: int,
        encoder_feedforward_dim: int,
        encoder_dropout: float,
        encoder_math_backend: bool,
        decoder_num_layers: int,
        decoder_share_projector: bool,
        decoder_dropout: float,
        reconstruct_all: bool,
        loss_func: str,
        loss_alpha: float,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.mask_ratio = mask_ratio
        self.reconstruct_all = reconstruct_all
        self.loss_func = loss_func
        self.loss_alpha = loss_alpha
        
        if mask_token_learnable:
            self.mask_token = nn.Parameter(torch.zeros(in_dim))
        else:
            self.register_buffer('mask_token', torch.zeros(in_dim))

        self.input_linear_list = nn.ModuleList([
            nn.Linear(in_dim, embedding_dim)
            for _ in range(sequence_len)
        ]) 
        
        self.encoder = Encoder(
            encoder_type = encoder_type,
            sequence_len = sequence_len,
            embedding_dim = embedding_dim,
            num_layers = encoder_num_layers,
            num_heads = encoder_num_heads,
            feedforward_dim = encoder_feedforward_dim,
            dropout = encoder_dropout,
            math_backend = encoder_math_backend,
        )
            
        self.decoder = Decoder(
            in_dim = embedding_dim,
            out_dim = in_dim,
            sequence_len = sequence_len,
            share_projector = decoder_share_projector,
            num_layers = decoder_num_layers,
            dropout = decoder_dropout,
        )
        
    def forward(
        self,
        node_feat: Tensor,
    ) -> Tensor:
        batch_size = len(node_feat)
        assert node_feat.shape == (batch_size, self.sequence_len, self.in_dim)

        mask = create_random_mask(
            size = self.sequence_len,
            mask_ratio = self.mask_ratio,
        )
        mask_cnt = mask.sum().item()
        assert mask.shape == (self.sequence_len,)
        
        masked_node_feat = node_feat.clone()
        masked_node_feat[:, mask, :] = self.mask_token.reshape(1, self.in_dim)

        h_list = [] 

        for i in range(self.sequence_len):
            h = self.input_linear_list[i]( masked_node_feat[:, i, :] )
            assert h.shape == (batch_size, self.embedding_dim)
            h_list.append(h)

        h = torch.stack(h_list, dim=1)
        assert h.shape == (batch_size, self.sequence_len, self.embedding_dim)
        
        embedding = self.encoder(h)
        assert embedding.shape == (batch_size, self.sequence_len, self.embedding_dim)
        
        recon = self.decoder(embedding)
        assert recon.shape == (batch_size, self.sequence_len, self.in_dim)
        
        if not self.reconstruct_all:
            recon = recon[:, mask, :]
            target = node_feat[:, mask, :]
            assert recon.shape == target.shape == (batch_size, mask_cnt, self.in_dim)
        else:
            target = node_feat 
            assert recon.shape == target.shape == (batch_size, self.sequence_len, self.in_dim)
        
        loss = compute_reconstruction_loss(
            input = recon,
            target = target,
            loss_func = self.loss_func,
            alpha = self.loss_alpha,
        )
        
        return loss 
        
    @torch.no_grad()
    def compute_node_embedding(
        self,
        node_feat: Tensor,
    ) -> Tensor:
        batch_size = len(node_feat)
        assert node_feat.shape == (batch_size, self.sequence_len, self.in_dim)

        h_list = [] 

        for i in range(self.sequence_len):
            h = self.input_linear_list[i]( node_feat[:, i, :] )
            assert h.shape == (batch_size, self.embedding_dim)
            h_list.append(h)

        h = torch.stack(h_list, dim=1)
        assert h.shape == (batch_size, self.sequence_len, self.embedding_dim)
        
        embedding = self.encoder(h)
        assert embedding.shape == (batch_size, self.sequence_len, self.embedding_dim)

        embedding = embedding.reshape(batch_size, self.sequence_len * self.embedding_dim)

        return embedding.detach() 
