from ..common import * 
from .logging import log_info 

_device: torch.device = torch.device('cpu') 


def get_device() -> torch.device:
    return _device 
    
    
def set_device(device: Any) -> torch.device: 
    global _device 
    _device = torch.device(device)
    
    log_info(f"The device is set as {_device}.")
    
    return _device 


def auto_select_gpus(max_cnt: int) -> list[torch.device]: 
    with os.popen('gpustat --json') as fp: 
        result_str = fp.read() 
        
    result_json = json.loads(result_str) 
    
    gpu_info_list = [] 
    
    for entry in result_json['gpus']:
        gpu_id = int(entry['index'])
        used_mem = int(entry['memory.used'])
        gpu_info_list.append((used_mem, gpu_id))
    
    gpu_info_list.sort() 
    
    device_list = [] 
    
    for i, entry in zip(range(max_cnt), gpu_info_list): 
        device = torch.device(f"cuda:{entry[1]}") 
        device_list.append(device) 
    
    return device_list 


def auto_set_device(use_gpu: bool = True) -> torch.device:
    if not use_gpu:
        return set_device('cpu')
    else:
        device = auto_select_gpus(max_cnt=1)[0]
        
        return set_device(device)  


def to_device(obj: Any,
              device: Any = None) -> Any:
    if device: 
        device = torch.device(device)
    else:
        device = get_device()
    
    if isinstance(obj, list):
        return [
            to_device(item, device=device)
            for item in obj 
        ]
    elif isinstance(obj, dict):
        return {
            key: to_device(value, device=device)
            for key, value in obj.items() 
        } 
    elif isinstance(obj, Tensor): 
        return obj.to(device) 
    elif isinstance(obj, ndarray):
        return torch.from_numpy(obj).to(device)
    elif isinstance(obj, dglsp.SparseMatrix): 
        return obj.to(device)
    elif hasattr(obj, 'to'): 
        return obj.to(device) 
    else:
        raise TypeError  
