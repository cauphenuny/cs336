import torch
from einops import rearrange, einsum
from jaxtyping import Float
from typing import Callable
from loguru import logger

def _backward_impl(grad_output, logsumexp, query, key, value, output, is_causal):
    dim = query.shape[-1]
    num_queries = query.shape[1]
    num_keys = key.shape[1]


    score = einsum(query, key, "b q d, b k d -> b q k") / (dim ** 0.5)
    if is_causal:
        mask = torch.arange(num_queries, device=query.device)[:, None] >= torch.arange(num_keys, device=query.device)[None, :]
        mask = rearrange(mask, "q k -> 1 q k")
        score = torch.where(mask, score, -1.0e6)

    prob = torch.exp(score - rearrange(logsumexp, "b q -> b q 1")) # b q k, grad_output: b q d
    rowsum = torch.sum(output * grad_output, dim=-1, keepdim=True) # rowsum(o * do) = rowsum(p * dp)
    grad_v = prob.mT @ grad_output
    grad_p = grad_output @ value.mT
    grad_s = prob * (grad_p - rowsum)
    grad_q = (grad_s @ key) / (dim ** 0.5)
    grad_k = (grad_s.mT @ query) / (dim ** 0.5)
    return grad_q, grad_k, grad_v, None

try:
    f_backward = torch.compile(_backward_impl)
except Exception as e:
    logger.error(f"Compilation failed with error: {e}")
    f_backward = _backward_impl