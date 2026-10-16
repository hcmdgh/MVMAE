from ..common import * 
from ..util.logging import log_debug, log_info


class Evaluator:
    def __init__(self,
                 early_stopping_metric: Optional[str] = None, 
                 early_stopping_epochs: Optional[int] = None, 
                 display_interval: int = 1, 
                 use_wandb: bool = False,
                 mute: bool = False):
        self.early_stopping_metric = early_stopping_metric 
        self.early_stopping_epochs = early_stopping_epochs 
        self.display_interval = display_interval
        self.use_wandb = use_wandb 
        self.mute = mute 
            
        self.record_dict: dict[str, dict[int, float]] = dict() 

        self.min_best_metric_set: set[str] = { 'Train Loss' }

    def record_train_loss(self,
                          epoch: int,
                          loss: Any):
        self.record_metric(
            epoch = epoch, 
            name = 'Train Loss', 
            value = loss, 
            min_best = True, 
        )
    
    def record_metric(self,
                      epoch: int, 
                      name: str, 
                      value: Any,
                      min_best: bool = False): 
        value = float(value) 
        
        if name not in self.record_dict: 
            self.record_dict[name] = dict() 
            
        if min_best: 
            self.min_best_metric_set.add(name) 
        
        assert epoch not in self.record_dict[name]
        self.record_dict[name][epoch] = value 
        
        if not min_best:
            best_epoch, best_value = max(self.record_dict[name].items(), key=lambda x: (x[1], -x[0]))

            if not self.mute and epoch % self.display_interval == 0:
                log_debug(f"[Epoch {epoch}] {name}: {value:.4f} (Max: {best_value:.4f} in Epoch {best_epoch}).")
        else:
            best_epoch, best_value = min(self.record_dict[name].items(), key=lambda x: (x[1], x[0]))
            
            if not self.mute and epoch % self.display_interval == 0:
                log_debug(f"[Epoch {epoch}] {name}: {value:.4f} (Min: {best_value:.4f} in Epoch {best_epoch}).")

        if self.use_wandb:
            wandb.log(
                { name: value }, 
                step = epoch, 
            )
            
    def check_early_stopping(self) -> bool:
        if not self.early_stopping_epochs or not self.early_stopping_metric or self.early_stopping_epochs <= 0: 
            return False 
        
        record_dict = self.record_dict[self.early_stopping_metric]
        
        current_epoch = max(record_dict.keys()) 
        previous_epoch = current_epoch - self.early_stopping_epochs  
        
        if previous_epoch <= 0:
            return False 
        
        assert previous_epoch in record_dict  
        
        previous_metric = record_dict[previous_epoch]
        history_metrics = [metric for epoch, metric in record_dict.items() if epoch >= previous_epoch]
        
        if (self.early_stopping_metric not in self.min_best_metric_set and max(history_metrics) <= previous_metric) \
                or (self.early_stopping_metric in self.min_best_metric_set and min(history_metrics) >= previous_metric):
            if not self.mute:
                log_info(f"[Early Stopping] Epoch: {current_epoch}")
             
            return True 
        else:
            return False 

    def summary(self) -> dict[str, float]:
        log_str = "[Summary]\n" 
        
        summary_dict: dict[str, float] = dict() 
        
        for metric_name in self.record_dict.keys(): 
            if metric_name in self.min_best_metric_set: 
                best_epoch, best_value = min(self.record_dict[metric_name].items(), key=lambda x: (x[1], x[0]))

                summary_dict[f"Min {metric_name}"] = best_value
                summary_dict[f"Min {metric_name} Epoch"] = best_epoch 

                log_str += f"    Min {metric_name}: {best_value:.4f} (in Epoch {best_epoch})\n"
                
                if self.use_wandb:        
                    wandb.summary[f"Min {metric_name}"] = best_value         
                    wandb.summary[f"Min {metric_name} Epoch"] = best_epoch     
                    
            else:
                best_epoch, best_value = max(self.record_dict[metric_name].items(), key=lambda x: (x[1], -x[0]))
                
                summary_dict[f"Max {metric_name}"] = best_value
                summary_dict[f"Max {metric_name} Epoch"] = best_epoch 

                log_str += f"    Max {metric_name}: {best_value:.4f} (in Epoch {best_epoch})\n"
                
                if self.use_wandb:        
                    wandb.summary[f"Max {metric_name}"] = best_value         
                    wandb.summary[f"Max {metric_name} Epoch"] = best_epoch       

        if not self.mute:
            log_info(log_str)

        if self.use_wandb:
            wandb.finish() 

        return summary_dict 
