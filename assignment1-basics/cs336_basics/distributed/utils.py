import torch.distributed as dist

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return not is_distributed() or dist.get_rank() == 0