from ..network.models import TransformerLM
from ..tokenize.tokenizer import Tokenizer
import torch


def generate(
    input_text: str,
    model: TransformerLM,
    tokenizer: Tokenizer,
    max_length: int = 2048,
    temperature: float = 1e-5,
    top_p: float = 0.9,
    end_token: str | bytes = b"<|endoftext|>",
) -> str:
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_ids, device=model.device)
    output = model.generate(
        input_tensor,
        end=tokenizer.token_id(end_token),
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
    )
    output_ids: list[int] = output.tolist()
    return tokenizer.decode(output_ids)

