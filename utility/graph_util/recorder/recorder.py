import torch 
from torch import Tensor 
import swanlab 
from swanlab.plugin.notification import WXWorkCallback
import os 
import json 
import copy 
import time 
from typing import Any, Optional


class Recorder:
    def __init__(
        self, 
        main_metric: str,
        early_stopping_patience: int,
        output_dir: str,
        optimize_direction: str = 'max',
        mute: bool = False,
        swanlab_enable: bool = False,
        swanlab_project: str = '',
        swanlab_exp_name: str = '',
        swanlab_config: Optional[dict] = None,
        swanlab_wechat_notify: bool = False,
    ):
        self.main_metric = main_metric 
        self.early_stopping_patience = early_stopping_patience
        self.optimize_direction = optimize_direction 
        self.mute = mute 
        self.output_dir = output_dir
        self.swanlab_enable = swanlab_enable 

        self.best_epoch_data = dict() 
        self.epoch_start_time = None 
        self.max_epoch = -1 

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.writer = open(os.path.join(output_dir, 'record.jsonl'), 'wt', encoding='utf-8')
        else:
            self.writer = None

        if swanlab_enable:
            assert swanlab_project
            
            callback_list = []

            if swanlab_wechat_notify:
                wechat_webhook_url = os.getenv('WECHAT_WEBHOOK_URL', '')
                assert wechat_webhook_url

                callback_list.append(
                    WXWorkCallback(
                        webhook_url = wechat_webhook_url,
                    )
                )

            swanlab.init(
                project = swanlab_project,
                experiment_name = swanlab_exp_name or None,
                config = swanlab_config,
                callbacks = callback_list,
            )

    def start_epoch(self):
        assert not self.epoch_start_time 
        self.epoch_start_time = time.perf_counter()

    def end_epoch(
        self,
        epoch: int,
        **epoch_data,
    ):
        assert epoch > self.max_epoch 
        self.max_epoch = epoch 
        
        epoch_data = {'epoch': epoch} | epoch_data 

        for key, value in epoch_data.items():
            if isinstance(value, Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.detach().cpu() 

            if isinstance(value, float):
                value = round(value, 5)

            epoch_data[key] = value

        assert self.epoch_start_time 
        epoch_data['duration'] = round(time.perf_counter() - self.epoch_start_time, 3)
        self.epoch_start_time = None 

        if not self.best_epoch_data:
            self.best_epoch_data = epoch_data
        else:
            best_metric = self.best_epoch_data[self.main_metric]
            current_metric = epoch_data[self.main_metric]

            if self.optimize_direction == 'max':
                if current_metric > best_metric:
                    self.best_epoch_data = epoch_data
            elif self.optimize_direction == 'min':
                if current_metric < best_metric:
                    self.best_epoch_data = epoch_data
            else:
                raise ValueError   
            
        if self.swanlab_enable:
            swanlab_epoch_data = {
                key: value 
                for key, value in epoch_data.items()
                if isinstance(value, (int, float))
            }

            swanlab.log(
                data = swanlab_epoch_data,
                step = epoch,
            )
            
        if self.writer:
            json_epoch_data = {
                key: value 
                for key, value in epoch_data.items()
                if isinstance(value, (int, float, str, list, tuple, dict))
            }
            json_str = json.dumps(json_epoch_data, ensure_ascii=False) 
            print(json_str, file=self.writer, flush=True)

        if not self.mute:
            print(epoch_data)

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience <= 0:
            return False 
        
        best_epoch = self.best_epoch_data['epoch'] 
        assert self.max_epoch >= best_epoch

        return self.max_epoch - best_epoch > self.early_stopping_patience  

    def summarize(self) -> dict:
        assert self.best_epoch_data

        if not self.mute:
            print() 
            print('[Summary]')
            print(f"max_epoch: {self.max_epoch}")
            print(f"best_epoch_data: {self.best_epoch_data}")
            print() 

        if self.swanlab_enable:
            swanlab_epoch_data = {
                f"final_{key}": value 
                for key, value in self.best_epoch_data.items()
                if isinstance(value, (int, float))
            }

            swanlab.log(
                data = swanlab_epoch_data,
            )

            swanlab.finish()

        if self.writer:
            self.writer.close()

        if self.output_dir:
            with open(os.path.join(self.output_dir, 'summary.json'), 'wt', encoding='utf-8') as w:
                json_epoch_data = {
                    key: value 
                    for key, value in self.best_epoch_data.items()
                    if isinstance(value, (int, float, str, list, tuple, dict))
                }

                json.dump(
                    json_epoch_data, 
                    w, 
                    ensure_ascii = False, 
                    indent = 4,
                )

        return self.best_epoch_data
