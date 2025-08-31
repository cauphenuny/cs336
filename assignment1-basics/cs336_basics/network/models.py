import torch
import tqdm
from .layers import Module, ModuleList
from .layers import RMSNorm, MultiheadSelfAttention, FeedForward, Embedding, Linear
from . import functional
from jaxtyping import Float, Int
from loguru import logger


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
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model,
            num_heads,
            rope_theta=rope_theta,
            rope_len=rope_len,
            device=device,
            dtype=dtype,
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
        share_embeddings: bool = False,
        device: torch.device | str | None = None,
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
        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
            weight=self.token_embeddings.weight if share_embeddings else None,
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

    def generate(
        self,
        input: Int[torch.Tensor, " seq_len"],
        end: int = 0,
        max_length: int = 2048,
        temperature: float = 1e-5,
        top_p=0.9,
    ) -> Int[torch.Tensor, " gen_len"]:
        self.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(range(max_length), desc="Generating")
            try:
                for _ in pbar:
                    logits = self(input)
                    probs = functional.softmax(logits[-1, :] / temperature, dim=-1)
                    next_token = functional.nucleus_sampling(probs, top_p)
                    input = torch.cat([input, next_token])
                    if next_token.item() == end:
                        break
            except KeyboardInterrupt:
                logger.info("Generation interrupted by user.")
            # pbar.
        return input
