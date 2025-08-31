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
    flush: bool = True,
):
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_ids, device=model.device)
    output_ids: list[int] = []
    for output_id in model.generate(
        input_tensor,
        end=tokenizer.token_id(end_token),
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        flush=flush,
    ):
        output_ids.append(output_id)
        try:
            yield tokenizer.decode(output_ids, errors="strict")
            output_ids = []
        except UnicodeDecodeError:
            pass
    if output_ids:
        yield tokenizer.decode(output_ids)
