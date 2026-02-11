from cs336_basics import network
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Int, Bool
from torch import Tensor
import einops


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    query: Float[Tensor, " ... len_q dim_k"],
    key: Float[Tensor, " ... len_k dim_k"],
    value: Float[Tensor, " ... len_k dim_v"],
    mask: Bool[Tensor, " ... len_q len_k"] | None = None,
) -> Float[Tensor, " ... len_q dim_v"]:
    with nvtx.range("computing attention score"):
        scores = einops.einsum(query, key, " ... len_q dim_k, ... len_k dim_k -> ... len_q len_k")
        scores = scores / key.shape[-1] ** 0.5
        if mask is not None:
            scores.masked_fill_(~mask, float("-inf"))
    with nvtx.range("computing softmax"):
        attn_value = network.functional.softmax(scores, dim=-1)
    with nvtx.range("computing final matmul"):
        result = einops.einsum(attn_value, value, " ... len_q len_k, ... len_k dim_v -> ... len_q dim_v")
    return result


original_ffn_forward = network.layers.FeedForward.forward


@nvtx.range("feed forward")
def annotated_ffn_forward(self, x):
    return original_ffn_forward(self, x)


network.layers.FeedForward.forward = annotated_ffn_forward
network.functional.scaled_dot_product_attention = annotated_scaled_dot_product_attention
