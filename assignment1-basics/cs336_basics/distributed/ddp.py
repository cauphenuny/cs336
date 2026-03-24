import torch
import torch.distributed as dist
import types
from loguru import logger
from .utils import is_distributed, is_main_process


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None = None):
        super().__init__()
        self.module = module
        self.async_handles = []
        self._sync_parameters()

        def _remove_requires_grad(_self, _x):
            raise NotImplementedError("DDP managed module can not set requires_grad")

        for submodule in self.modules():
            submodule.requires_grad_ = types.MethodType(_remove_requires_grad, submodule)

        for param in self.parameters():
            param.requires_grad_ = types.MethodType(_remove_requires_grad, param)

        world_size = dist.get_world_size() if is_distributed() else 1

        if bucket_size_mb is None:

            def grad_hook(x):
                handle = dist.all_reduce(
                    x.grad, op=dist.ReduceOp.SUM, async_op=True)

                def callback():
                    x.grad /= dist.get_world_size()

                self.async_handles.append((handle, callback))
                return x

            for param in self.module.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(grad_hook)

        else:
            buckets = []
            bucket = []
            current_size = 0
            for param in reversed(list(self.module.parameters())):
                if not param.requires_grad:
                    continue
                size = param.numel() * param.element_size() / (1024 * 1024)
                if current_size + size > bucket_size_mb:
                    buckets.append(bucket)
                    bucket = []
                    current_size = 0
                current_size += size
                bucket.append(param)
            if bucket:
                buckets.append(bucket)

            logger.info(
                f"DDP initialized with {len(buckets)} buckets, bucket size: {bucket_size_mb} MB")
            self.complete_count = [0] * len(buckets)

            for bid, bucket in enumerate(buckets):

                def grad_hook(x, bid=bid, bucket=bucket):
                    self.complete_count[bid] += 1
                    if self.complete_count[bid] == len(bucket):
                        flatten = torch._utils._flatten_dense_tensors(
                            [p.grad for p in bucket])
                        handle = dist.all_reduce(
                            flatten, op=dist.ReduceOp.SUM, async_op=True)

                        def callback(world_size=world_size, bucket=bucket, flatten=flatten):
                            grads = torch._utils._unflatten_dense_tensors(
                                flatten, [p.grad for p in bucket])
                            for x, g in zip(bucket, grads):
                                x.grad.copy_(g / world_size)

                        self.async_handles.append((handle, callback))
                        self.complete_count[bid] = 0
                    return x

                for param in bucket:
                    param.register_post_accumulate_grad_hook(grad_hook)

    def _wait(self):
        for handle, callback in self.async_handles:
            handle.wait()
            callback()
        self.async_handles = []

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _sync_parameters(self):
        for p in self.module.parameters():
            if is_distributed():
                dist.broadcast(p, src=0)

    def finish_gradient_synchronization(self):
        self._wait()


class DistributedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
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

        dataset_size = len(self.dataset)  # type: ignore
        if self.drop_last:
            self.num_samples = dataset_size // self.world_size
        else:
            self.num_samples = (
                dataset_size + self.world_size - 1) // self.world_size
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))  # type: ignore
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(
                len(self.dataset), generator=g).tolist()  # type: ignore

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
