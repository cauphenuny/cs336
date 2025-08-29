import torch
from torch import nn, Tensor
from jaxtyping import Float, Bool, Int
import einops


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def swiglu(
    x: Float[Tensor, " ... d_model"],
    weight_1: Float[Tensor, " d_ff d_model"],
    weight_2: Float[Tensor, " d_model d_ff"],
    weight_3: Float[Tensor, " d_ff d_model"],
) -> Float[Tensor, " ... d_model"]:
    return (silu(x @ weight_1.T) * (x @ weight_3.T)) @ weight_2.T


def softmax(x: Tensor, dim: int = -1):
    maximum = torch.max(x, dim=dim, keepdim=True).values
    x -= maximum
    x = torch.exp(x)
    x /= torch.sum(x, dim=dim, keepdim=True)
    return x


def scaled_dot_product_attention(
    query: Float[Tensor, " ... len_q dim_k"],
    key: Float[Tensor, " ... len_k dim_k"],
    value: Float[Tensor, " ... len_k dim_v"],
    mask: Bool[Tensor, " ... len_q len_k"] | None = None,
) -> Float[Tensor, " ... len_q dim_v"]:
    scores = einops.einsum(
        query, key, " ... len_q dim_k, ... len_k dim_k -> ... len_q len_k"
    )
    scores /= key.shape[-1] ** 0.5
    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))
    attn_value = softmax(scores, dim=-1)
    return einops.einsum(
        attn_value, value, " ... len_q len_k, ... len_k dim_v -> ... len_q dim_v"
    )


def cross_entropy(
    logits: Float[Tensor, " ... batch classes"], targets: Int[Tensor, " ... batch"]
) -> Float[Tensor, " ..."]:
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    log_probs = logits - max_logits
    log_probs = -(
        log_probs - torch.log(torch.sum(torch.exp(log_probs), dim=-1, keepdim=True))
    )
    entropy: Float[Tensor, " ... batch"] = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return entropy.mean(dim=-1)


def perplexity(
    logits: Float[Tensor, " ... batch seq_len vocab_size"],
    targets: Int[Tensor, " ... batch seq_len"],
):
    logits = einops.rearrange(logits, "... batch len vocab-> ... len batch vocab")
    targets = einops.rearrange(targets, "... batch len -> ... len batch")
    ce: Float[Tensor, " ... len"] = cross_entropy(logits, targets)
    return ce.mean(dim=-1)
