import argparse
import os
from loguru import logger
import wandb
from jaxtyping import Float
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from ..network.models import TransformerLM
from ..tokenize.tokenizer import Tokenizer
from ..optimize.optimizers import AdamW
from ..network.multiplatform import ACCL_DEVICE
from ..network import functional
from .dataset import TextDataLoader, TextDataset
from .checkpoint import save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser()

parser.add_argument("dataset", type=str)
parser.add_argument("--output", type=str, default="outputs")
parser.add_argument("--project", type=str, default="CS336 - Assignment 1")
parser.add_argument("--name", type=str, default="experiment")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--val_interval", type=int, default=200)

parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000.0)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--max_train_tokens", type=int, default=327_680_000)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=0.01)


def main():
    args = parser.parse_args()
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project=args.project,
        name=args.name,
        config=vars(args),
    )

    device = ACCL_DEVICE
    model_args = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        device=device,
    )
    model = TransformerLM(**model_args)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    train_loader = TextDataLoader(
        path=os.path.join(args.dataset, f"train-{args.vocab_size}.npy"),
        context_length=args.context_length,
        batch_size=args.batch_size,
        limit=args.max_train_tokens,
        limit_type="total_tokens",
        vocab_size=args.vocab_size,
        # device=device,
    )
    val_loader = TextDataLoader(
        path=os.path.join(args.dataset, f"valid-{args.vocab_size}.npy"),
        context_length=args.context_length,
        batch_size=args.batch_size,
        limit=20,
        limit_type="train_steps",
        vocab_size=args.vocab_size,
        # device=device,
    )

    best_loss = float("inf")
    best_perplexity = float("inf")
    train_path = os.path.join(args.output, args.name + f"-{run.id}")
    os.makedirs(train_path, exist_ok=True)
    checkpoint_path = os.path.join(train_path, "checkpoint.pt")
    best_checkpoint_path = os.path.join(train_path, "best_checkpoint.pt")

    def validate():
        nonlocal best_loss, best_perplexity
        model.eval()
        with torch.no_grad():
            vlosses = []
            vperps = []
            pbar = tqdm(val_loader, desc="Validation")
            for input, target in pbar:
                input, target = input.to(device), target.to(device)
                output_logits = model(input)
                loss = functional.cross_entropy(output_logits, target).mean()
                perplexity = functional.perplexity(output_logits, target).mean()
                vlosses.append(loss.item())
                vperps.append(perplexity.item())
                pbar.set_postfix(v_loss=f"{loss.item():.3f}", v_perplexity=f"{perplexity.item():.3f}")
            vloss = float(np.mean(vlosses))
            vperp = float(np.mean(vperps))
            wandb.log(
                {"val_loss": vloss, "val_perplexity": vperp},
                step=step,
            )
        save_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, iter=step, model_args=model_args, run_id=run.id
        )
        if vloss < best_loss:
            best_loss = vloss
            best_perplexity = vperp
            save_checkpoint(
                best_checkpoint_path, model=model, optimizer=optimizer, iter=step, model_args=model_args, run_id=run.id
            )
        model.train()
        return vloss, vperp

    pbar = tqdm(train_loader)

    for step, (input, target) in enumerate(pbar):
        optimizer.zero_grad()
        # logger.debug(f"Train Step {step}: {input.shape = }, {target.shape = }, {input.dtype = }")
        input, target = input.to(device), target.to(device)
        output_logits: Float[Tensor, " ... batch len vocab_size"] = model(input)
        loss = functional.cross_entropy(output_logits, target).mean()
        loss.backward()
        optimizer.step()
        if step % args.log_interval == 0:
            wandb.log(
                {"train_loss": loss.item()},
                step=step,
            )
        pbar.set_postfix(loss=f"{loss.item():.3f}")
        if (step + 1) % args.val_interval == 0:
            validate()
    validate()


if __name__ == "__main__":
    main()
