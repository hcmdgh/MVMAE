import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor 
from dataclasses import dataclass
from tqdm import tqdm  
from typing import Optional, Any, Callable

from graph_util.recorder.recorder import Recorder


@dataclass(kw_only=True)
class LinearProbe:
    in_dim: int 
    hidden_dim: int
    out_dim: int
    main_metric_name: str 
    metric_func_dict: dict[str, Callable]
    optimize_direction: str = 'max'
    num_layers: int = 1 
    lr: float = 0.001
    num_epochs: int = 1000
    early_stopping_patience: int = 30
    use_tqdm: bool = False 

    def train(
        self,
        train_feat: Tensor,
        train_label: Tensor,
        val_feat: Tensor,
        val_label: Tensor,
        test_feat: Tensor,
        test_label: Tensor,
    ) -> dict:
        device = train_feat.device
        train_cnt = len(train_feat)
        val_cnt = len(val_feat)
        test_cnt = len(test_feat)
        assert train_feat.shape == (train_cnt, self.in_dim)
        assert val_feat.shape == (val_cnt, self.in_dim)
        assert test_feat.shape == (test_cnt, self.in_dim)
        assert train_label.shape == (train_cnt,)
        assert val_label.shape == (val_cnt,)
        assert test_label.shape == (test_cnt,)

        if self.num_layers == 1:
            model = nn.Linear(self.in_dim, self.out_dim).to(device)
        elif self.num_layers == 2:
            model = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            ).to(device)
        else:
            raise ValueError 

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        recorder = Recorder(
            main_metric = 'val_' + self.main_metric_name,
            optimize_direction = self.optimize_direction,
            early_stopping_patience = self.early_stopping_patience,
            output_dir = '',
            mute = True,
        )
        
        for epoch in tqdm(range(self.num_epochs), desc='LinearProbe', disable=not self.use_tqdm):
            recorder.start_epoch()
            model.train() 

            train_logit = model(train_feat)
            
            train_loss = F.cross_entropy(input=train_logit, target=train_label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            model.eval() 
            
            with torch.no_grad():
                val_logit = model(val_feat)

                test_logit = model(test_feat)

            val_metric_dict = {
                metric_name: metric_func(
                    input = val_logit,
                    target = val_label,
                )
                for metric_name, metric_func in self.metric_func_dict.items()
            }

            test_metric_dict = {
                metric_name: metric_func(
                    input = test_logit,
                    target = test_label,
                )
                for metric_name, metric_func in self.metric_func_dict.items()
            }
                
            recorder.end_epoch(
                epoch = epoch,
                train_loss = train_loss,
                **{
                    f'val_{metric_name}': metric_value 
                    for metric_name, metric_value in val_metric_dict.items()
                },
                **{
                    f'test_{metric_name}': metric_value 
                    for metric_name, metric_value in test_metric_dict.items()
                },
            )

            if recorder.check_early_stopping():
                break
                
        linear_probe_result = recorder.summarize()
        print(f"{linear_probe_result = }\n")

        linear_probe_result = {
            k: v 
            for k, v in linear_probe_result.items()
            if k.startswith('val_') or k.startswith('test_')
        }

        return linear_probe_result 
