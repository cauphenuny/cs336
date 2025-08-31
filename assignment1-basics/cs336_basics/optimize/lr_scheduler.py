from . import functional
import torch


class CosineLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.05,
        anneal_ratio: float = 1,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.anneal_steps = int(anneal_ratio * total_steps)
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

    def update(self, current_step):
        lr: list[float] | float = []
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = functional.lr_cosine_schedule(
                current_step, self.initial_lrs[i], self.min_lr, self.warmup_steps, self.anneal_steps
            )
            lr.append(group["lr"])
        if len(lr) == 1:
            lr = lr[0]
        return lr
