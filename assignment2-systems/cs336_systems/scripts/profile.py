import cs336_systems
import os
import argparse
import pandas as pd
from pathlib import Path


def main(args):
    presets = {
        "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
        "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    }
    hyperparams = presets[args.preset]

    if args.output:
        parent = Path(args.output).parent
        os.makedirs(parent, exist_ok=True)

    for context_length in [128, 256, 512, 1024]:
        path = args.output + f"-context{context_length}.pickle" if args.output else None
        mean, std = cs336_systems.cuda.benchmark(
            dict(**hyperparams, context_length=context_length),
            n_warmup=args.n_warmup,
            n_step=args.n_step,
            backward=False,
            profile_memory=path,
        )
        print(f"context length: {context_length}, {args.n_warmup} warmup, {args.n_step} step:")
        print(f"mean: {mean:.6f} sec, std: {std:.6f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, default="output/profile_memory", help="output file to save profile results"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["small", "medium", "large", "xl", "2.7B"],
        default="2.7B",
        help="Model preset to benchmark",
    )
    parser.add_argument("--n_warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--n_step", type=int, default=10, help="Number of benchmark iterations")
    args = parser.parse_args()
    main(args)
