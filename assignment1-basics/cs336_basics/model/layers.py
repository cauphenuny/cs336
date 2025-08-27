import torch
import einops
from jaxtyping import Float
from torch import nn, Tensor


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Embedding Block

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            d_model (int): Dimension of the embedding vectors (d_model).
            device (torch.device | None): Device to place the embeddings on (default: None).
            dtype (torch.dtype | None): Data type of the embeddings (default: None).
        """
        super().__init__()
        self.weights: Float[Tensor, " vocab_size d_model"] = nn.Parameter(
            torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            token_ids (Tensor): Tensor of token IDs to be embedded.

        Returns:
            Tensor: Embedded representations of the input token IDs.
        """
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    """
    RMSNorm Module
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Root Mean Square Layer Normalization

        Args:
            d_model (int): Dimension of the model (number of features).
            eps (float): Small value to avoid division by zero.
            device (torch.device | None): Device to place the parameters on (default: None).
            dtype (torch.dtype | None): Data type of the parameters (default: None).
        """
        super().__init__()
        self.weights: Float[Tensor, " d_model"] = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.device = device

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Tensor:
        """
        Args:
            x (Tensor(shape=(..., d_model))): input
        Returns:
            RMS normalized tensor with the same shape as input
        """
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)  # upcast to prevent overflow
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)
        result = x_f32 / rms * self.weights
        return result.to(in_dtype)


class FeedForward(nn.Module):
    """
    SwiGLU Position-Wise Feed-Forward Layer
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # d_ff = d_ff if d_ff else
        pass


class RoPE(nn.Module):
    pass
