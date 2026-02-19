import torch
from einops import rearrange, einsum
from jaxtyping import Float
from typing import Callable
from loguru import logger

def _backward_impl(grad_output, logsumexp, query, key, value, output, is_causal):
    # Compute in fp32 to support bf16 inputs and improve numerical stability.
    # (Mixed-dtype matmuls like fp32 @ bf16 error out in PyTorch/Inductor.)
    dim = query.shape[-1]
    num_queries = query.shape[1]
    num_keys = key.shape[1]

    qf = query.float()
    kf = key.float()
    vf = value.float()
    gof = grad_output.float()
    of = output.float()

    scale = 1.0 / (dim ** 0.5)
    score = einsum(qf, kf, "b q d, b k d -> b q k") * scale
    if is_causal:
        mask = torch.arange(num_queries, device=query.device)[:, None] >= torch.arange(num_keys, device=query.device)[None, :]
        mask = rearrange(mask, "q k -> 1 q k")
        score = torch.where(mask, score, -1.0e6)

    lse = rearrange(logsumexp.float(), "b q -> b q 1")
    prob = torch.exp(score - lse)  # [b, q, k]

    # rowsum(o * do) = rowsum(p * dp)
    rowsum = torch.sum(of * gof, dim=-1, keepdim=True)

    grad_v = prob.mT @ gof
    grad_p = gof @ vf.mT
    grad_s = prob * (grad_p - rowsum)
    grad_q = (grad_s @ kf) * scale
    grad_k = (grad_s.mT @ qf) * scale

    return grad_q.to(dtype=query.dtype), grad_k.to(dtype=key.dtype), grad_v.to(dtype=value.dtype), None

try:
    f_backward = torch.compile(_backward_impl)
except Exception as e:
    logger.error(f"Compilation failed with error: {e}")
    f_backward = _backward_impl