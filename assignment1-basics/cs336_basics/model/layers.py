import torch
from jaxtyping import Float
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight: Float[torch.Tensor, "d_out d_in"] = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        # TODO: Adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight @ x


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Embedding Block

        Args:
            num_embeddings (int): Number of unique tokens in the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors (d_model).
            device (torch.device | None): Device to place the embeddings on (default: None).
            dtype (torch.dtype | None): Data type of the embeddings (default: None).
        """
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs to be embedded.

        Returns:
            torch.Tensor: Embedded representations of the input token IDs.
        """
        raise NotImplementedError


class RMSNorm(nn.Module):
    pass


class FeedForward(nn.Module):
    pass


class RotaryPositionalEmbedding(nn.Module):
    pass
