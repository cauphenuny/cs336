import os
from loguru import logger
import torch
import typing
import glob
from termcolor import colored
import questionary

from ..network.multiplatform import ACCL_DEVICE

from ..network.models import TransformerLM


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
    model = TransformerLM(**model_args)
    model.load_state_dict(checkpoint["model"])
    return model


def find_models(path: str = ".", pattern="*model.pt"):
    pt_files = glob.glob(os.path.join(path, "**", pattern), recursive=True)
    models: dict[str, tuple[int, float, float]] = {}
    for file in pt_files:
        try:
            checkpoint = torch.load(file, map_location="cpu")
            models[file] = (
                checkpoint["iter"],
                checkpoint.get("loss", float("inf")),
                checkpoint.get("perplexity", float("inf")),
            )
            torch.save(checkpoint, file)
        except Exception:
            pass
    sorted_models = sorted(models.items(), key=lambda x: (x[1][2], x[1][1], -x[1][0]))
    return sorted_models


def select_model(path: str = ".", pattern: str = "*model.pt", verbose: bool = True):
    models = find_models(path, pattern)
    choices = []
    ppl_thresh = [0, 8, 20]
    for model in models:
        ppl = model[1][2]
        if ppl > ppl_thresh[2]:
            color = "red"
        elif ppl > ppl_thresh[1]:
            color = "yellow"
        else:
            color = "green"
        name = f"Model: {model[0]}, iter: {model[1][0]}, loss: {model[1][1]:.3f}, perplexity: {colored(f'{model[1][2]:.3f}', color)}"
        if verbose:
            logger.info(name)
        choices.append(
            {
                "name": f"{model[0]}, iter={model[1][0]}, ppl={model[1][2]:.3f}",
                "value": model[0],
            }
        )

    choice = questionary.select("Choose a model:", choices=choices).ask()
    return choice
