from collections.abc import Callable, Iterable
from typing import Optional 
import torch 
import math 

class SGD(torch.optim.Optimizer): 
    def __init__(self, params, lr: float = 1e-3, lr_decay: bool = True): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}") 
        defaults = {"lr": lr, "lr_decay": lr_decay} 
        super().__init__(params, defaults) 

    def step(self, closure: Callable | None = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups: 
            lr = group["lr"] # Get the learning rate.
            lr_decay = group["lr_decay"] # Get the weight decay.
            for p in group["params"]: 
                if p.grad is None: 
                    continue 
                state = self.state[p] # Get state associated with p. 
                grad = p.grad.data # Get the gradient of loss with respect to p. 
                if lr_decay:
                    t = state.get("t", 0) # Get iteration number from the state, or initial value. 
                    p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
                    state["t"] = t + 1 # Increment iteration number. 
                else:
                    p.data -= lr * grad # Update weight tensor in-place without decay.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], eps: float, weight_decay: float):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                # Adam update
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 0) + 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                lr_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                state["m"] = m
                state["v"] = v
                state["t"] = t
                # Apply weight decay
                if weight_decay > 0:
                    p.data -= weight_decay * lr * p.data
        return loss