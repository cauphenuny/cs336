import torch
from torch import nn, Tensor
from jaxtyping import Float, Bool
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
