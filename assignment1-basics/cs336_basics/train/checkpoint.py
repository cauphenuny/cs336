import os
from loguru import logger
import torch
import typing
import glob
from termcolor import colored
import questionary
import tarfile
import tempfile

from ..network.multiplatform import ACCL_DEVICE
from ..network.models import TransformerModel
from ..tokenize.tokenizer import Tokenizer


def save_checkpoint(
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iter: int,
    model_args: dict,
    **kwargs,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iter,
        "model_args": model_args,
        **kwargs,
    }
    torch.save(checkpoint, out)


def save_model(
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    iter: int,
    model_args: dict,
    **kwargs,
):
    checkpoint = {
        "model": model.state_dict(),
        "iter": iter,
        "model_args": model_args,
        **kwargs,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def load_model(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint: dict = torch.load(src)
    model_args: dict = checkpoint.get(
        "model_args",
        {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 512,
            "d_ff": 1344,
            "rope_theta": 10000.0,
            "num_heads": 16,
            "num_layers": 4,
            "device": ACCL_DEVICE,
        },
    )
    logger.info(f"Loaded model with args: {model_args}")
    perplexity = checkpoint.get("perplexity", checkpoint.get("best_perplexity", None))
    if perplexity:
        logger.info(f"Model perplexity: {perplexity}")
    model = TransformerModel(**model_args)
    model.load_state_dict(checkpoint["model"])
    return model


def format_model_metrics(name: str, iter: int, loss: float, perplexity: float, train_tokens: float) -> str:
    ppl_thresh = [0, 8, 20]
    if perplexity > ppl_thresh[2]:
        color = "red"
    elif perplexity > ppl_thresh[1]:
        color = "yellow"
    else:
        color = "green"
    return f"Model: {name}, train_tokens: {train_tokens / 1024 / 1024:.3f}M (iter={iter}), loss: {loss:.4f}, perplexity: {colored(f'{perplexity:.4f}', color)}"


def find_models(path: str = ".", pattern: str = "*model.pt", verbose: bool = False):
    model_files = glob.glob(os.path.join(path, "**", pattern), recursive=True)
    models: dict[str, tuple[int, float, float, float]] = {}
    for file in model_files:
        try:
            checkpoint = torch.load(file, map_location="cpu")
            models[file] = (
                checkpoint["iter"],
                checkpoint.get("loss", float("inf")),
                checkpoint.get("perplexity", float("inf")),
                checkpoint.get("train_tokens", float("nan")),
            )
        except Exception:
            pass
    sorted_models = sorted(models.items(), key=lambda x: (x[1][2], x[1][1], -x[1][0]))
    for model in sorted_models:
        if verbose:
            logger.info(format_model_metrics(model[0], *model[1]))
    return sorted_models


def select_model(path: str = ".", pattern: str = "*model.pt", verbose: bool = True) -> str:
    models = find_models(path, pattern, verbose=verbose)
    choices = []
    for model in models:
        choices.append(
            {
                "name": f"{model[0]}, iter={model[1][0]}, ppl={model[1][2]:.3f}",
                "value": model[0],
            }
        )

    choice = questionary.select("Choose a model:", choices=choices).ask()
    return choice


class TransformerLM:
    def __init__(self, model_file: str | os.PathLike | typing.IO[bytes], tokenizer: Tokenizer):
        self.model_file = model_file
        self.model = load_model(model_file)
        self.tokenizer = tokenizer

    def generate(
        self,
        input_text: str,
        max_length: int = 2048,
        temperature: float = 1e-5,
        top_p: float = 0.9,
        end_token: str | bytes = b"<|endoftext|>",
        flush: bool = True,
    ):
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor(input_ids, device=self.model.device)
        output_ids: list[int] = []
        for output_id in self.model.generate(
            input_tensor,
            end=self.tokenizer.token_id(end_token),
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            flush=flush,
        ):
            output_ids.append(output_id)
            try:
                yield self.tokenizer.decode(output_ids, errors="strict")
                output_ids = []
            except UnicodeDecodeError:
                pass
        if output_ids:
            yield self.tokenizer.decode(output_ids)

    def save(self, path: str | os.PathLike, base_name: str):
        filepath = os.path.join(path, base_name + ".tar.gz")
        with tarfile.open(filepath, "w:gz") as tar:
            if isinstance(self.model_file, str | os.PathLike):
                tar.add(self.model_file, arcname="model.pt")
            else:
                raise NotImplementedError("Saving from file-like object is not supported yet.")
            with tempfile.NamedTemporaryFile(delete=True) as vocab:
                with tempfile.NamedTemporaryFile(delete=True) as merges:
                    self.tokenizer.to_files(vocab.name, merges.name)
                    tar.add(vocab.name, arcname="tokenizer-vocab.json")
                    tar.add(merges.name, arcname="tokenizer-merges.json")

    @classmethod
    def load(cls, path: str | os.PathLike):
        with tarfile.open(path, "r:gz") as tar:
            model_file = tar.extractfile("model.pt")
            if not model_file:
                raise ValueError("Failed to extract model.pt")
            tokenizer_vocab = tar.extractfile("tokenizer-vocab.json")
            tokenizer_merges = tar.extractfile("tokenizer-merges.json")
            if not tokenizer_vocab or not tokenizer_merges:
                raise ValueError("Failed to extract tokenizer files")
            return TransformerLM(model_file, Tokenizer.from_files(tokenizer_vocab, tokenizer_merges))

