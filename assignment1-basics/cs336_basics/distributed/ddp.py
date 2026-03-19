import torch
import torch.distributed as dist
from .utils import is_distributed, is_main_process

def sync(grad):
    if is_distributed():
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad /= dist.get_world_size()
    return grad

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, register: bool = True):
        super().__init__()
        self.module = module
        self.sync_parameters()
        if register:
            self.register_hook()
        self.registered = register

    def register_hook(self, hook=lambda x: sync(x)):
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_hook(hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_parameters(self):
        for p in self.module.parameters():
            if is_distributed():
                dist.broadcast(p, src=0)

    def sync_gradients(self, force: bool = False):
        if self.registered and not force:
            return

        for p in self.module.parameters():
            if p.grad is not None:
                sync(p.grad)

class DistributedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, shuffle: bool = True, drop_last: bool = False):
        if not hasattr(dataset, "__len__"):
            raise TypeError(
                "DistributedSampler requires a map-style dataset with __len__. "
                "For IterableDataset, shard data in dataset.__iter__."
            )
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        if is_distributed():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        dataset_size = len(self.dataset) # type: ignore
        if self.drop_last:
            self.num_samples = dataset_size // self.world_size
        else:
            self.num_samples = (dataset_size + self.world_size - 1) // self.world_size
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        indices = list(range(len(self.dataset))) # type: ignore
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist() # type: ignore

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch