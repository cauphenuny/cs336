import torch
from .layers import Module, ModuleList
from .layers import RMSNorm, MultiheadSelfAttention, FeedForward, Embedding, Linear
from . import functional
from jaxtyping import Float, Int


class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_theta: float | None = None,
        rope_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, rope_theta=rope_theta, rope_len=rope_len, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, " ... seq_len d_model"]):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        rope_theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff=d_ff,
                    rope_theta=rope_theta,
                    rope_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, x: Int[torch.Tensor, " ... seq_len"], softmax: bool = False
    ) -> Float[torch.Tensor, " ... seq_len vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        if softmax:
            x = functional.softmax(x, dim=-1)
        return x
