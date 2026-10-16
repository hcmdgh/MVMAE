import os 
import sys 
sys.path.append(os.path.expanduser('~/utility'))

import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import basic_util 
import graph_util 

from .model import MVMAE  
from .trainer import Trainer
from .propagate import propagate_along_metapaths


class Arguments(basic_util.BaseArguments):
    dataset: str 
    split: int 
    metapath_schema: str 

    normalize_node_feat: bool 
    normalize_adj_mat: str 
    embedding_dim: int 
    mask_ratio: float 
    mask_token_learnable: bool
    encoder_type: str
    encoder_num_layers: int 
    encoder_num_heads: int 
    encoder_feedforward_dim: int 
    encoder_dropout: float
    encoder_math_backend: bool 
    decoder_num_layers: int 
    decoder_share_projector: bool
    decoder_dropout: float
    reconstruct_all: bool 
    loss_func: str
    loss_alpha: float
    linear_probe_num_layers: int 

    swanlab_enable: bool 
    swanlab_project: str  
    swanlab_exp_name: str  
    swanlab_wechat_notify: bool 
    seed: int 
    lr: float 
    weight_decay: float 
    num_epochs: int 
    early_stopping_patience: int 
    output_dir: str 


def main():
    args = Arguments.parse_args()
    basic_util.json_dump(dict(args), os.path.join(args.output_dir, 'argument.json'))
    print(f"{args = }\n")

    if args.dataset == 'ACM':
        dataset = graph_util.HeCoACMDataset(
            split = args.split,
            metapath_schema = args.metapath_schema,
        ).cuda() 
    elif args.dataset == 'DBLP':
        dataset = graph_util.HeCoDBLPDataset(
            split = args.split,
            metapath_schema = args.metapath_schema,
        ).cuda() 
    elif args.dataset == 'Freebase':
        dataset = graph_util.HeCoFreebaseDataset(
            split = args.split,
            metapath_schema = args.metapath_schema,
            node_feat_init = 'randn_256',
        ).cuda() 
    elif args.dataset == 'AMiner':
        dataset = graph_util.HINAMinerDataset(
            split = args.split,
            metapath_schema = args.metapath_schema,
            node_feat_init = 'randn_256',
        ).cuda() 
    elif args.dataset == 'RCDD_subgraph':
        dataset = graph_util.RCDDSubgraphDataset(
            metapath_schema = args.metapath_schema,
        ).cuda() 
    elif args.dataset == 'RCDD':
        dataset = graph_util.RCDDDataset(
            metapath_schema = args.metapath_schema,
        ).cuda() 
    else:
        raise ValueError

    if args.normalize_node_feat:
        dataset.normalize_node_feat_()

    prop_node_feat_list = propagate_along_metapaths(
        edge_index_dict = dataset.edge_index_dict(),
        node_feat_dict = dataset.node_feat_dict(),
        metapath_list = dataset.metapath_list,
        normalize_adj_mat = args.normalize_adj_mat,
    )

    prop_node_feat = torch.stack(prop_node_feat_list, dim=1).cuda() 

    num_metapaths = len(dataset.metapath_list)
    print(f"{num_metapaths = }\n")

    model = MVMAE(
        in_dim = dataset.target_node_feat_dim(),
        embedding_dim = args.embedding_dim,
        sequence_len = num_metapaths,
        mask_ratio = args.mask_ratio,
        mask_token_learnable = args.mask_token_learnable,
        encoder_type = args.encoder_type,
        encoder_num_layers = args.encoder_num_layers,
        encoder_num_heads = args.encoder_num_heads,
        encoder_feedforward_dim = args.encoder_feedforward_dim,
        encoder_dropout = args.encoder_dropout,
        encoder_math_backend = args.encoder_math_backend,
        decoder_num_layers = args.decoder_num_layers,
        decoder_share_projector = args.decoder_share_projector,
        decoder_dropout = args.decoder_dropout,
        reconstruct_all = args.reconstruct_all,
        loss_func = args.loss_func,
        loss_alpha = args.loss_alpha,
    ).cuda() 

    trainer = Trainer(
        model = model,
        main_metric_name = dataset.main_metric_name(),
        optimize_direction = 'max',
        seed = args.seed,
        optimizer = 'adam',
        lr = args.lr,
        weight_decay = args.weight_decay,
        num_epochs = args.num_epochs,
        early_stopping_patience = args.early_stopping_patience, 
        output_dir = args.output_dir,
        swanlab_enable = args.swanlab_enable,
        swanlab_project = args.swanlab_project,
        swanlab_exp_name = args.swanlab_exp_name,
        swanlab_config = dict(args),
        swanlab_wechat_notify = args.swanlab_wechat_notify, 
    )    

    trainer.set_linear_probe(
        graph_util.LinearProbe(
            in_dim = args.embedding_dim * num_metapaths,
            hidden_dim = args.embedding_dim,
            out_dim = dataset.num_classes(),
            main_metric_name = dataset.main_metric_name(),
            metric_func_dict = dataset.metric_func_dict(),
            optimize_direction = 'max',
            num_layers = args.linear_probe_num_layers, 
            lr = 0.001,
            num_epochs = 1000,
            early_stopping_patience = 30,
            use_tqdm = False,
        )
    )
    
    trainer.train(
        train_data = dict(
            node_feat = prop_node_feat, 
        ),
        eval_data = dict(
            node_feat = prop_node_feat, 
            node_label = dataset.target_node_label(),
            train_mask = dataset.train_mask(),
            val_mask = dataset.val_mask(),
            test_mask = dataset.test_mask(),
        ),
    )


if __name__ == '__main__':
    main() 
