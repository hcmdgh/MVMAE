import torch 
import torch.nn as nn
from torch import Tensor 
from torch.utils.data import DataLoader
from dataclasses import dataclass
import os 
from typing import Callable, Optional, Any 

from graph_util.seed import set_seed 
from graph_util.recorder.recorder import Recorder
from graph_util.memory import print_gpu_and_memory_usage


@dataclass(kw_only=True)
class Trainer:
    model: nn.Module
    main_metric_name: str
    optimize_direction: str = 'max'
    seed: int = 42
    optimizer: str = 'adam'
    lr: float 
    weight_decay: float = 0.
    num_epochs: int 
    eval_interval: int = 1
    early_stopping_patience: int = -1 
    output_dir: str
    swanlab_enable: bool = False 
    swanlab_project: str = 'default'
    swanlab_exp_name: str = ''
    swanlab_config: Optional[dict] = None 
    swanlab_wechat_notify: bool = True 

    def __post_init__(self):
        CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
        print(f"{CUDA_VISIBLE_DEVICES = }\n")

        set_seed(self.seed)

        self.model = self.model.cuda()

        try:
            num_trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"{num_trainable_parameters = }\n")
        except ValueError:
            pass 

        if self.optimizer == 'adam':
            self.optimizer_obj = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError
        
        self.recorder = Recorder(
            main_metric = f"val_{self.main_metric_name}",
            optimize_direction = self.optimize_direction,
            early_stopping_patience = self.early_stopping_patience,
            output_dir = self.output_dir,
            swanlab_enable = self.swanlab_enable,
            swanlab_project = self.swanlab_project,
            swanlab_exp_name = self.swanlab_exp_name,
            swanlab_config = self.swanlab_config,
            swanlab_wechat_notify = self.swanlab_wechat_notify,
        )

    def train_epoch(
        self,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError

    def evaluate_epoch(
        self,
        **kwargs,
    ) -> dict:
        raise NotImplementedError

    def train(
        self,
        train_data: dict,
        eval_data: dict,
    ):
        for epoch in range(self.num_epochs):
            self.recorder.start_epoch()
            self.model.train()

            loss = self.train_epoch(**train_data)

            self.optimizer_obj.zero_grad()
            loss.backward() 
            self.optimizer_obj.step()

            self.model.eval()

            if (epoch + 1) % self.eval_interval == 0:
                eval_metric_dict = self.evaluate_epoch(**eval_data)
            else:
                eval_metric_dict = dict() 

            self.recorder.end_epoch(
                epoch = epoch,
                train_loss = loss,
                **eval_metric_dict,
            )

            if self.recorder.check_early_stopping():
                break 

            print_gpu_and_memory_usage()

        self.recorder.summarize()
