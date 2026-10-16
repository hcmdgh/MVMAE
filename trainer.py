import torch 
import torch.nn.functional as F 
from torch import Tensor
from typing import Any 

import graph_util 


class Trainer(graph_util.Trainer):
    def set_linear_probe(
        self,
        linear_probe: graph_util.LinearProbe,
    ):
        self.linear_probe = linear_probe

    def train_epoch(
        self, 
        node_feat: Tensor,
    ) -> Tensor:
        loss = self.model(
            node_feat = node_feat,
        )

        return loss 
    
    def evaluate_epoch(
        self,
        node_feat: Tensor,
        node_label: Tensor,
        train_mask: Tensor,
        val_mask: Tensor,
        test_mask: Tensor,
    ) -> dict:
        node_embedding = self.model.compute_node_embedding(
            node_feat = node_feat,
        )

        result = self.linear_probe.train(
            train_feat = node_embedding[train_mask],
            train_label = node_label[train_mask],
            val_feat = node_embedding[val_mask],
            val_label = node_label[val_mask],
            test_feat = node_embedding[test_mask],
            test_label = node_label[test_mask],
        )

        return result 
