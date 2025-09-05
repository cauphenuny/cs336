import argparse
import os
import sys
from loguru import logger
import wandb
from jaxtyping import Float
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from typing import Literal

from ..network.models import TransformerModel
from .. import optimize
from ..optimize.optimizers import AdamW
from ..optimize.lr_scheduler import CosineLRScheduler
from ..network.multiplatform import ACCL_DEVICE, profile, compile_backend
from ..network import functional
from .dataset import TextDataLoader, TextDataset
from .checkpoint import save_checkpoint, load_checkpoint, save_model

parser = argparse.ArgumentParser()

parser.add_argument("--profile", action="store_true", default=False)
parser.add_argument("--compile", action="store_true", default=False)
parser.add_argument("--max_epoch", type=int)
parser.add_argument("--check_dataset", action="store_true", default=False)

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output", type=str, default="outputs")
parser.add_argument("--project", type=str, default="CS336 - Assignment 1")
parser.add_argument("--name", type=str, default="experiment")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--val_interval", type=int, default=500)
parser.add_argument("--val_sample", type=int, default=20)

parser.add_argument(
    "--model_preset", type=str, choices=["nano", "micro", "tiny", "small", "medium", "large", "huge", "ultimate"]
)
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000.0)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=4)


def convert_to_bool(value: str) -> bool:
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser.add_argument("--share_embeddings", type=convert_to_bool, default=False)
parser.add_argument("--max_train_tokens", type=int, default=327_680_000)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=0.01)


def load_preset(preset: Literal["nano", "micro", "tiny", "small", "medium", "large", "huge", "ultimate"]):
    presets = {
        "nano": dict(
            d_model=64, d_ff=256, num_heads=2, num_layers=4, batch_size=64, lr=1e-3, max_train_tokens=40_960_000
        ),
        "micro": dict(
            d_model=128, d_ff=320, num_heads=4, num_layers=4, batch_size=64, lr=1e-3, max_train_tokens=40_960_000
        ),
        "tiny": dict(
            d_model=256, d_ff=640, num_heads=8, num_layers=4, batch_size=32, lr=1e-3, max_train_tokens=81_920_000
        ),
        "small": dict(
            d_model=512, d_ff=1344, num_heads=16, num_layers=4, batch_size=32, lr=1e-3, max_train_tokens=327_680_000
        ),
        "medium": dict(
            d_model=512, d_ff=1344, num_heads=16, num_layers=8, batch_size=16, lr=8e-4, max_train_tokens=655_360_000
        ),
        "large": dict(
            d_model=768, d_ff=2048, num_heads=16, num_layers=8, batch_size=16, lr=8e-4, max_train_tokens=655_360_000
        ),
        "huge": dict(
            d_model=1024,
            d_ff=2888,
            num_heads=16,
            num_layers=16,
            context_length=512,
            batch_size=8,
            lr=5e-4,
            max_train_tokens=1_310_720_000,
        ),
        "ultimate": dict(
            d_model=1536,
            d_ff=4096,
            num_heads=16,
            num_layers=24,
            context_length=512,
            batch_size=8,
            lr=5e-4,
            max_train_tokens=2_621_440_000,
        ),
    }
    for p in ("nano", "micro", "tiny"):
        presets[p]["share_embeddings"] = True
    for p in ("small", "medium", "large", "huge", "ultimate"):
        presets[p]["share_embeddings"] = False

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}")
    args = presets[preset]
    for k, v in args.items():
        if parser.get_default(k) != v:
            logger.info(f"Setting {k} to {v} (was {parser.get_default(k)})")
            parser.set_defaults(**{k: v})


def main():
    args = parser.parse_args()
    if args.model_preset is not None:
        load_preset(args.model_preset)
        args = parser.parse_args()
    logger.info(f"Arguments: {vars(args)}")
    device = ACCL_DEVICE
    logger.info(f"Train on: {device}")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    model_args = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        share_embeddings=args.share_embeddings,
        device=device,
    )
    model = TransformerModel(**model_args)
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
        limit=args.val_sample,
        limit_type="train_steps",
        vocab_size=args.vocab_size,
        # device=device,
    )
    lr_scheduler = CosineLRScheduler(
        optimizer,
        total_steps=len(train_loader),
        min_lr=args.min_lr,
    )

    if args.check_dataset:
        train_loader.check()
        val_loader.check()

    run_id = None
    start_iter = 0
    best_loss = float("inf")
    best_perplexity = float("inf")

    if args.resume is not None:
        if os.path.isfile(args.resume):
            ckpt = load_checkpoint(args.resume, model=model, optimizer=optimizer)
            run_id = ckpt.get("run_id", None)
            start_iter = ckpt.get("iter", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            best_perplexity = ckpt.get("best_perplexity", float("inf"))
            logger.info(f"Resumed from checkpoint {args.resume} at iteration {start_iter}.")
            start_iter += 1
            train_loader.set_start_iter(start_iter)
            if start_iter >= len(train_loader):
                logger.error(f"Start iteration {start_iter} exceeds total training steps {len(train_loader)}.")
                exit(1)
        else:
            logger.error(f"Checkpoint file {args.resume} does not exist.")

    run = wandb.init(
        project=args.project,
        name=args.name,
        config=vars(args),
        id=run_id,
        resume="allow",
    )

    train_path = os.path.join(args.output, args.name + f"-{run.id}")
    os.makedirs(train_path, exist_ok=True)
    checkpoint_path = os.path.join(train_path, "checkpoint.pt")
    best_model_path = os.path.join(train_path, "best_model.pt")
    trace_path = os.path.join(train_path, "trace")

    def validate():
        nonlocal best_loss, best_perplexity
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                vlosses = []
                vperps = []
                for input, target in pbar:
                    input, target = input.to(device), target.to(device)
                    output_logits = model(input)
                    loss = functional.cross_entropy(output_logits, target).mean()
                    perplexity = functional.perplexity(output_logits, target).mean()
                    vlosses.append(loss.item())
                    vperps.append(perplexity.item())
                    vloss = float(np.mean(vlosses))
                    vperp = float(np.mean(vperps))
                    pbar.set_postfix(v_loss=f"{vloss:.3f}", v_perplexity=f"{vperp:.3f}")
            wandb.log(
                {"val_loss": vloss, "val_perplexity": vperp},
                step=step,
            )
            outputs = [checkpoint_path]
            if vloss < best_loss:
                best_loss = vloss
                best_perplexity = vperp
                save_model(
                    best_model_path,
                    model=model,
                    iter=step + 1,
                    model_args=model_args,
                    run_id=run.id,
                    best_loss=best_loss,
                    best_perplexity=best_perplexity,
                    loss=vloss,
                    perplexity=vperp,
                    train_tokens=(step + 1) * args.batch_size * args.context_length,
                )
                outputs.append(best_model_path)
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                iter=step,
                model_args=model_args,
                run_id=run.id,
                best_loss=best_loss,
                best_perplexity=best_perplexity,
                loss=vloss,
                perplexity=vperp,
                train_tokens=(step + 1) * args.batch_size * args.context_length,
            )
            print("\r\033[K", file=sys.stderr, end="")  # clear line
            logger.info(
                f"Saved checkpoint to {', '.join(outputs)}. loss: {vloss:.3f}/{best_loss:.3f}, perplexity: {vperp:.3f}/{best_perplexity:.3f}"
            )
        model.train()
        return vloss, vperp

    if args.compile:
        model.compile(backend=compile_backend())

    with profile(enable=args.profile, json_trace_file=trace_path if args.profile else None) as prof:
        try:
            epoch_cnt = 0
            with tqdm(train_loader, initial=start_iter, total=len(train_loader), desc="Training", unit="step") as pbar:
                train_loss = float("nan")
                grad = float("nan")
                for step, (input, target) in enumerate(pbar, start=start_iter):
                    current_lr = lr_scheduler.update(step)
                    optimizer.zero_grad()
                    # logger.debug(f"Train Step {step}: {input.shape = }, {target.shape = }, {input.dtype = }")
                    input, target = input.to(device), target.to(device)
                    output_logits: Float[Tensor, " ... batch len vocab_size"] = model(input)
                    loss = functional.cross_entropy(output_logits, target).mean()
                    loss.backward()
                    optimize.functional.gradient_clip(model.parameters(), 2.0)
                    if step % args.log_interval == 0:
                        grad = optimize.functional.gradient_norm(model.parameters()).cpu()
                        train_loss = loss.cpu()
                        wandb.log(
                            {"train_loss": train_loss, "grad": grad, "lr": current_lr},
                            step=step,
                        )
                        pbar.set_postfix(loss=f"{train_loss:.3f}", grad=f"{grad:.3e}", lr=f"{current_lr:.3e}")
                    optimizer.step()
                    pbar.set_postfix(loss=f"{train_loss:.3f}", grad=f"{grad:.3e}", lr=f"{current_lr:.3e}")
                    if prof:
                        prof.step()
                    if (step + 1) % args.val_interval == 0:
                        validate()
                        epoch_cnt += 1
                        if args.max_epoch and epoch_cnt >= args.max_epoch:
                            logger.info(f"Reached max epoch {args.max_epoch}, stopping training.")
                            break
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        else:
            validate()


if __name__ == "__main__":
    main()
