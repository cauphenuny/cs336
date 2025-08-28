import torch
from torch import nn, Tensor
from jaxtyping import Float
import einops


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def swiglu(
    x: Float[Tensor, " ... d_model"],
    weight_1: Float[Tensor, " d_model d_ff"],
    weight_2: Float[Tensor, " d_model d_ff"],
    weight_3: Float[Tensor, " d_model d_ff"],
) -> Float[Tensor, " ... d_model"]:
    return einops.einsum(
        weight_2,
        silu(x @ weight_3) * x @ weight_1,
        "d_model d_ff, ... d_ff -> d_model ...",
    )
