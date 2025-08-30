from typing import override
from sympy import Mul
import torch
import einops
from jaxtyping import Float, Int
from torch import Tensor
from . import functional


class Module(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = "unknown"

    def param_size(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self):
        param = next(self.parameters(), None)
        if param is not None:
            self._device = param.device
        return self._device

    @device.setter
    def device(self, value: torch.device | str | None):
        self._device = value


class ModuleList(torch.nn.ModuleList, Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.device = device
        self.dtype = dtype
        self.weight: Float[Tensor, "d_out d_in"] = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Embedding Block

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            d_model (int): Dimension of the embedding vectors (d_model).
            device (torch.device | str | None): Device to place the embeddings on (default: None).
            dtype (torch.dtype | None): Data type of the embeddings (default: None).
        """
        super().__init__()
        self.weight: Float[Tensor, " vocab_size d_model"] = torch.nn.Parameter(
            torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            token_ids (Tensor): Tensor of token IDs to be embedded.

        Returns:
            Tensor: Embedded representations of the input token IDs.
        """
        return self.weight[token_ids]


class RMSNorm(Module):
    """
    RMSNorm Module
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Root Mean Square Layer Normalization

        Args:
            d_model (int): Dimension of the model (number of features).
            eps (float): Small value to avoid division by zero.
            device (torch.device | str | None): Device to place the parameters on (default: None).
            dtype (torch.dtype | None): Data type of the parameters (default: None).
        """
        super().__init__()
        self.weight: Float[Tensor, " d_model"] = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        # self.device = device

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
        result = x_f32 / rms * self.weight
        return result.to(in_dtype)


class FeedForward(Module):
    """
    SwiGLU Position-Wise Feed-Forward Layer
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        def auto_dim_ff(d_model, align: int = 64):
            raw_dim = int(8 * d_model / 3)
            return ((raw_dim + align - 1) // align) * align

        d_ff = d_ff if d_ff else auto_dim_ff(d_model)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]):
        return functional.swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)


class RoPE(Module):
    rotate_x: Float[Tensor, " max_seq_len d_k"]
    rotate_y: Float[Tensor, " max_seq_len d_k"]
    freqs: Float[Tensor, " half_d_k"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int = 0,
        device: torch.device | str | None = None,
    ):
        """
        Args:
            theta: theta value for RoPE
            d_k: dimension of query and key vectors
            max_seq_len: maximum sequence length that will be inputted
        """
        super().__init__()
        self.device = device
        freqs = torch.pow(theta, -torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        self.register_buffer("freqs", freqs, persistent=False)
        self.max_seq_len = max_seq_len
        if max_seq_len:
            self._update_rotation(max_seq_len)
            self.dynamic_expand = False
        else:
            self.dynamic_expand = True

    def _update_rotation(self, max_seq_len: int):
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = einops.einsum(positions, self.freqs, "max_seq_len, half_d_k -> max_seq_len half_d_k")
        """
        [   cos     -sin    ] @ [x] = [cos x - sin y]
        [   sin     cos     ]   [y]   [sin x + cos y]
        """
        rotate_x: Float[Tensor, "max_seq_len d_k"] = torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=-1
        ).flatten(start_dim=-2)
        rotate_y: Float[Tensor, "max_seq_len d_k"] = torch.stack(
            [-torch.sin(angles), torch.cos(angles)], dim=-1
        ).flatten(start_dim=-2)
        if self.device:
            rotate_x = rotate_x.to(self.device)
            rotate_y = rotate_y.to(self.device)
        self.register_buffer("rotate_x", rotate_x, persistent=False)
        self.register_buffer("rotate_y", rotate_y, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"] | None,
    ) -> Float[Tensor, " ... seq_len d_k"]:
        if token_positions is None:
            token_positions = torch.arange(input.shape[-2], device=input.device)
        if self.dynamic_expand and torch.max(token_positions) > self.max_seq_len:
            self._update_rotation(int(torch.max(token_positions)))
        rot_x: Float[Tensor, " ... seq_len d_k"] = self.rotate_x[token_positions]
        rot_y: Float[Tensor, " ... seq_len d_k"] = self.rotate_y[token_positions]
        x: Float[Tensor, " ... seq_len d_k"] = input[..., 0::2].repeat_interleave(2, dim=-1)
        y: Float[Tensor, " ... seq_len d_k"] = input[..., 1::2].repeat_interleave(2, dim=-1)
        return (rot_x * x) + (rot_y * y)


class MultiheadSelfAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        casual: bool = True,
        rope_theta: float | None = None,
        rope_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.casual = casual
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(rope_theta, self.head_dim, rope_len, device=device) if rope_theta and rope_len else None

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = (
            einops.rearrange(
                tensor,
                "... len (num_heads head_dim) -> ... num_heads len head_dim",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
            for tensor in (q, k, v)
        )
        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        if self.casual:
            seq_len = x.shape[-2]
            mask = torch.arange(seq_len).reshape(-1, 1) >= torch.arange(seq_len)
            # print(mask)
            mask = mask.to(self.device)
        else:
            mask = None
        attn_output = functional.scaled_dot_product_attention(q, k, v, mask=mask)
        attn_output = einops.rearrange(
            attn_output,
            "... num_heads len dim -> ... len (num_heads dim)",
            num_heads=self.num_heads,
        )
        return self.output_proj(attn_output)
