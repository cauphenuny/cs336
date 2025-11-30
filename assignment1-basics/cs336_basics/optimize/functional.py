import math
import torch
from collections.abc import Iterable


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    it -= warmup_iters
    cosine_cycle_iters -= warmup_iters
    if it > cosine_cycle_iters:
        it = cosine_cycle_iters
    cosine_decay = 0.5 * (1 + math.cos(math.pi * it / cosine_cycle_iters))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay


def gradient_norm(params: Iterable[torch.nn.Parameter]):
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = torch.stack([g.norm() for g in grads]).norm()
    return total_norm


@torch.no_grad()
def gradient_clip(parameters, max_l2_norm: float, eps: float = 1e-6):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.)

    norms = torch.stack([g.norm() for g in grads])
    total_norm = torch.norm(norms)

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for g in grads:
        g.mul_(clip_coef_clamped)

    return total_norm
