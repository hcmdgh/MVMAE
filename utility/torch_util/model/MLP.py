from ..common import * 
from ..util.device import get_device
from ..dataloader.index_loader import IndexLoader 
from ..util.logging import log_warning 
from ..metric.acc import compute_acc
from ..metric.f1 import compute_f1_micro, compute_f1_macro
from ..evaluator.evaluator import Evaluator


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 batch_norm: bool = False, 
                 activation: nn.Module = nn.ELU(),
                 dropout: float = 0.):
        super().__init__() 
        
        self.in_dim = in_dim 
        self.hidden_dim = (in_dim + out_dim) // 2 
        self.out_dim = out_dim
        self.num_layers = num_layers 
        
        self.seq = nn.Sequential() 
        
        if num_layers == 1:
            self.seq.append(nn.Linear(in_dim, out_dim)) 
        elif num_layers >= 2:
            for l in range(num_layers):
                if l == 0:
                    self.seq.append(nn.Linear(in_dim, self.hidden_dim))
                elif l < num_layers - 1:
                    self.seq.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                else:
                    self.seq.append(nn.Linear(self.hidden_dim, out_dim)) 
                    
                if batch_norm and l < num_layers - 1:
                    self.seq.append(nn.BatchNorm1d(self.hidden_dim)) 
                    
                if l < num_layers - 1: 
                    self.seq.append(activation) 
                    
                if l < num_layers - 1: 
                    self.seq.append(nn.Dropout(dropout)) 
        else:
            raise AssertionError 

    def forward(self,
                src: FloatTensor) -> FloatTensor: 
        out = self.seq(src)  
        
        return out 
                     
    @staticmethod
    def classify(
        train_feat: FloatTensor,
        train_label: IntTensor,
        val_feat: FloatTensor,
        val_label: IntTensor,
        test_feat: FloatTensor,
        test_label: IntTensor,
        device: Any = None, 
        num_layers: int = 1,
        activation: nn.Module = nn.ReLU(),
        lr: float = 0.001,
        num_epochs: int = 500,
        batch_size: int = INF, 
        use_tqdm: bool = True,
        converge_warning: bool = True,  
    ) -> dict[str, Any]:
        raise DeprecationWarning 
        
        if device is None:
            device = get_device() 
        else:
            device = torch.device(device) 
        
        train_feat = train_feat.detach().float().to(device) 
        train_label = train_label.detach().long().to(device) 
        val_feat = val_feat.detach().float().to(device) 
        val_label = val_label.detach().long().to(device) 
        test_feat = test_feat.detach().float().to(device) 
        test_label = test_label.detach().long().to(device) 
        
        feat_dim = train_feat.shape[-1]
        num_classes = int(torch.max(torch.cat([train_label, val_label, test_label]))) + 1 

        model = MLP(
            in_dim = feat_dim, 
            out_dim = num_classes, 
            num_layers = num_layers, 
            activation = activation, 
        )
        model = model.to(device)
        
        train_loader = IndexLoader(
            len(train_feat), 
            batch_size = batch_size, 
            shuffle = True, 
        )
        val_loader = IndexLoader(
            len(val_feat), 
            batch_size = batch_size, 
            shuffle = False, 
        )
        test_loader = IndexLoader(
            len(test_feat), 
            batch_size = batch_size, 
            shuffle = False, 
        )

        real_lr = lr / len(train_loader)
        optimizer = optim.Adam(model.parameters(), lr=real_lr) 
        
        epoch_to_val_acc: dict[int, float] = dict() 
        epoch_to_test_acc: dict[int, float] = dict()

        for epoch in tqdm(range(1, num_epochs + 1), disable=not use_tqdm, desc='MLP Classify', unit='epoch'):
            model.train() 
            
            for batch_idx in train_loader:
                pred_batch = model(train_feat[batch_idx]) 
                label_batch = train_label[batch_idx] 
                
                loss = F.cross_entropy(input=pred_batch, target=label_batch)
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
            
            model.eval() 
            
            pred_batch_list = []
            label_batch_list = []
            
            for batch_idx in val_loader:
                with torch.no_grad():
                    pred_batch = model(val_feat[batch_idx]).detach().cpu()  
                    
                label_batch = val_label[batch_idx].detach().cpu() 
                
                pred_batch_list.append(pred_batch) 
                label_batch_list.append(label_batch)

            val_acc = compute_acc(
                pred = torch.cat(pred_batch_list, dim=0), 
                target = torch.cat(label_batch_list, dim=0), 
            )
            
            pred_batch_list = []
            label_batch_list = []
            
            for batch_idx in test_loader:
                with torch.no_grad():
                    pred_batch = model(test_feat[batch_idx]).detach().cpu()  
                    
                label_batch = test_label[batch_idx].detach().cpu() 
                
                pred_batch_list.append(pred_batch) 
                label_batch_list.append(label_batch)

            test_acc = compute_acc(
                pred = torch.cat(pred_batch_list, dim=0), 
                target = torch.cat(label_batch_list, dim=0), 
            )
            
            epoch_to_val_acc[epoch] = val_acc 
            epoch_to_test_acc[epoch] = test_acc 

        best_val_acc_epoch, best_val_acc = max(epoch_to_val_acc.items(), key=lambda x: (x[1], -x[0]))
        best_test_acc_epoch, best_test_acc = max(epoch_to_test_acc.items(), key=lambda x: (x[1], -x[0]))

        if best_val_acc_epoch >= num_epochs - 10 \
                or best_test_acc_epoch >= num_epochs - 10:
            if converge_warning:
                log_warning("The MLP model has not fully converged.")

        return dict(
            best_val_acc_epoch = best_val_acc_epoch,
            best_val_acc = best_val_acc,
            best_test_acc_epoch = best_test_acc_epoch,
            best_test_acc = best_test_acc,
        )

    
def train_and_eval_MLP(
        train_feat: FloatTensor,
        train_label: IntTensor,
        test_feat: FloatTensor,
        test_label: IntTensor,
        num_layers: int = 3,
        activation: Callable = nn.ReLU(),
        lr: float = 0.001,
        num_epochs: int = 2000,
        batch_size: int = -1, 
        early_stopping_epochs: int = 100, 
        device: Any = None, 
        use_tqdm: bool = True,
        converge_warning: bool = True, 
    ) -> dict[str, Any]:
        if not device:
            device = train_feat.device 

        train_feat = train_feat.detach()
        train_label = train_label.detach()
        test_feat = test_feat.detach() 
        test_label = test_label.detach() 
        
        if batch_size > 0:
            train_feat = train_feat.cpu()
            train_label = train_label.cpu()
            test_feat = test_feat.cpu() 
            test_label = test_label.cpu() 
        else:
            train_feat = train_feat.to(device)
            train_label = train_label.to(device)
            test_feat = test_feat.to(device)
            test_label = test_label.to(device)
        
        feat_dim = train_feat.shape[-1]
        num_classes = int(torch.max(torch.cat([train_label, test_label]))) + 1 
        
        model = MLP(
            in_dim = feat_dim, 
            out_dim = num_classes, 
            num_layers = num_layers, 
            activation = activation, 
        ).to(device) 
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        evaluator = Evaluator(
            early_stopping_epochs = early_stopping_epochs, 
            early_stopping_metric = 'Test F1-Micro', 
            use_wandb = False, 
            mute = True, 
        )

        for epoch in tqdm(range(1, num_epochs + 1), disable=not use_tqdm, desc='MLP Classify', unit='epoch'):
            model.train() 
            
            if batch_size <= 0:
                pred = model(train_feat)
                
                loss = F.cross_entropy(input=pred, target=train_label)
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
                
                model.eval() 
                
                with torch.no_grad():
                    test_pred = model(test_feat).detach_() 

                test_f1_micro = compute_f1_micro(pred=test_pred, target=test_label)
                test_f1_macro = compute_f1_macro(pred=test_pred, target=test_label)
                
                evaluator.record_metric(epoch=epoch, name='Test F1-Micro', value=test_f1_micro) 
                evaluator.record_metric(epoch=epoch, name='Test F1-Macro', value=test_f1_macro) 
            else:
                train_loader = DataLoader(
                    dataset = TensorDataset(train_feat, train_label), 
                    batch_size = batch_size, 
                    shuffle = True, 
                    drop_last = False, 
                )
                
                for train_feat_batch, train_label_batch in train_loader:
                    train_feat_batch = train_feat_batch.to(device) 
                    train_label_batch = train_label_batch.to(device) 
                    
                    pred_batch = model(train_feat_batch)
                
                    loss = F.cross_entropy(input=pred_batch, target=train_label_batch)
                
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step() 
                
                model.eval() 
                
                test_loader = DataLoader(
                    dataset = TensorDataset(test_feat, test_label), 
                    batch_size = batch_size, 
                    shuffle = False, 
                    drop_last = False, 
                )
                
                pred_batch_list = [] 
                label_batch_list = []
                
                for test_feat_batch, test_label_batch in test_loader:
                    test_feat_batch = test_feat_batch.to(device) 

                    with torch.no_grad():
                        pred_batch = model(test_feat_batch)
                    
                    pred_batch_list.append(pred_batch.detach().cpu())
                    label_batch_list.append(test_label_batch.cpu()) 

                pred = torch.cat(pred_batch_list, dim=0)
                target = torch.cat(label_batch_list, dim=0)
                
                test_f1_micro = compute_f1_micro(pred=pred, target=target)
                test_f1_macro = compute_f1_macro(pred=pred, target=target)
                
                evaluator.record_metric(epoch=epoch, name='Test F1-Micro', value=test_f1_micro) 
                evaluator.record_metric(epoch=epoch, name='Test F1-Macro', value=test_f1_macro) 
                
            if evaluator.check_early_stopping():
                break  

        result = evaluator.summary()
        max_test_f1_micro = result['Max Test F1-Micro']
        max_test_f1_micro_epoch = result['Max Test F1-Micro Epoch']
        max_test_f1_macro = result['Max Test F1-Macro']
        max_test_acc = max_test_f1_micro

        if max_test_f1_micro_epoch >= num_epochs - 10:
            if converge_warning:
                log_warning("The MLP model has not fully converged.")

        return dict(
            max_test_f1_micro = max_test_f1_micro,
            max_test_f1_micro_epoch = max_test_f1_micro_epoch,
            max_test_f1_macro = max_test_f1_macro,
            max_test_acc = max_test_acc, 
        )
