import torch 
import psutil 
import os 


def print_gpu_and_memory_usage():
    cuda_max_memory_allocated_dict = dict()

    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        cuda_max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
        cuda_max_memory_allocated_str = f"{cuda_max_memory_allocated / 1024 / 1024 / 1024:.3f}GB"

        if cuda_max_memory_allocated > 1024:
            cuda_max_memory_allocated_dict[device] = cuda_max_memory_allocated_str

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory = f"{memory_info.rss / 1024 / 1024 / 1024:.3f}GB"

    print(f"GPU usage: {cuda_max_memory_allocated_dict}")
    print(f"Memory usage: {memory}")
    print() 
