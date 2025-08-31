import os
import torch
import typing

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
    model = TransformerLM(**model_args)
    model.load_state_dict(checkpoint["model"])
    return model
