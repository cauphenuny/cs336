import torch
import tqdm
from . import functional
from .layers import Module, ModuleList
from .layers import RMSNorm, MultiheadSelfAttention, FeedForward, Embedding, Linear, Identical
from ..tokenize.tokenizer import Tokenizer
from jaxtyping import Float, Int
from loguru import logger
from typing import Literal


class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_theta: float | None = None,
        rope_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        norm_type: Literal["rms", "none"] = "rms",
        norm_location: Literal["pre", "post"] = "pre",
        ffn_activate: Literal["swiglu", "silu"] = "swiglu",
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if norm_type == "rms" else Identical()
        self.attn = MultiheadSelfAttention(
            d_model,
            num_heads,
            rope_theta=rope_theta,
            rope_len=rope_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if norm_type == "rms" else Identical()
        self.ffn = FeedForward(d_model, d_ff, activate=ffn_activate, device=device, dtype=dtype)
        self.norm_location = norm_location

    def forward(self, x: Float[torch.Tensor, " ... seq_len d_model"]):
        if self.norm_location == "pre":
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        elif self.norm_location == "post":
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        else:
            raise NotImplementedError(f"unsupported norm_location: {self.norm_location}")
        return x


class TransformerModel(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        rope_theta: float | None = None,
        share_embeddings: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        norm_type: Literal["rms", "none"] = "rms",
        norm_location: Literal["pre", "post"] = "pre",
        ffn_activate: Literal["swiglu", "silu"] = "swiglu",
    ):
        super().__init__()

        self.context_length = context_length
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
                    norm_type=norm_type,
                    norm_location=norm_location,
                    ffn_activate=ffn_activate,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm_type == "rms" else Identical()
        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
            weight=(self.token_embeddings.weight, True) if share_embeddings else None,
        )

        transformer_param = self.layers.param_size + self.ln_final.param_size
        embedding_param = self.param_size - transformer_param
        # logger.info(f"{self.lm_head.param_size = }")
        # logger.info(f"{self.token_embeddings.param_size = }")
        # logger.info(f"{embedding_param = }")
        logger.info(
            f"Model initialized with {self.param_size / 1024 / 1024:,.2f}M parameters"
            f"({embedding_param / 1024 / 1024:,.2f}M embedding, {transformer_param / 1024 / 1024:,.2f}M transformer)."
        )

    def forward(self, x: Int[torch.Tensor, " ... seq_len"]) -> Float[torch.Tensor, " ... seq_len vocab_size"]:
        assert x.dtype in (torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool, torch.long)
        x = x.to(torch.long)
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    def embed(self, input: Int[torch.Tensor, " seq_len"]) -> Float[torch.Tensor, " seq_len d_model"]:
        assert input.dtype in (
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
            torch.long,
        )
        input = input.to(torch.long)
        return self.token_embeddings(input)

    def generate(
        self,
        input: Int[torch.Tensor, " seq_len"],
        end: int = 0,
        max_length: int = 2048,
        temperature: float = 1e-5,
        top_p: float = 0.9,
        flush: bool = True,
    ):
        self.eval()
        output = torch.tensor([], device=input.device)
        with torch.no_grad():
            pbar = range(max_length)
            if not flush:
                pbar = tqdm.tqdm(range(max_length), desc="Generating")
            try:
                for _ in pbar:
                    logits = self(input)
                    probs = functional.softmax(logits[-1, :] / temperature, dim=-1)
                    next_token = functional.nucleus_sampling(probs, top_p)
                    # output = torch.cat([output, next_token])
                    if flush:
                        yield int(next_token.item())
                    else:
                        output = torch.cat([output, next_token])
                    if input.numel() < self.context_length:
                        input = torch.cat([input, next_token])
                    else:
                        input = torch.cat([input[1:], next_token])
                    if next_token.item() == end:
                        break
            except KeyboardInterrupt:
                logger.info("Generation interrupted by user.")
        if not flush:
            yield from output.tolist()
        # return output


def specifications(size: str):
    presets = {
        "nano": dict(d_model=64, num_heads=2, num_layers=4),
        "micro": dict(d_model=128, num_heads=4, num_layers=4),
        "tiny": dict(d_model=256, num_heads=8, num_layers=4),
        "small": dict(d_model=512, num_heads=16, num_layers=4),
        "medium": dict(d_model=512, num_heads=16, num_layers=8),
        "large": dict(d_model=768, num_heads=16, num_layers=8),
        "x-large": dict(d_model=1024, num_heads=16, num_layers=16),
        "xx-large": dict(d_model=1536, num_heads=16, num_layers=24),
        "3x-large": dict(d_model=2048, num_heads=16, num_layers=32),
        "4x-large": dict(d_model=2560, num_heads=20, num_layers=40),
        "5x-large": dict(d_model=3072, num_heads=24, num_layers=48),
    }
    for name, args in presets.items():
        args["share_embeddings"] = True if name in ("nano", "micro", "tiny") else False
    return presets[size]
