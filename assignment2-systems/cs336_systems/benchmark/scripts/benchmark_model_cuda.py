from cs336_systems.benchmark.cuda import benchmark
from loguru import logger
import torch
import argparse
import pandas as pd


def main(args):
    presets = {
        "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
        "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    }
    models: dict[str, dict[str, int]] = {}
    for name in args.models:
        models[name] = presets[name]
    context_length = args.context_length
    enable_compile = args.compile
    dtype = eval(f"torch.{args.dtype}")

    def run_benchmark(n_warmup, n_step):
        names, means, stds = [], [], []
        for name, args in models.items():
            try:
                mean, std = benchmark(
                    dict(**args, context_length=context_length),
                    n_warmup=n_warmup,
                    n_step=n_step,
                    backward=True,
                    compile=enable_compile,
                    dtype=dtype,
                )
            except Exception as e:
                logger.error(f"Error benchmarking model {name}: {e}")
                mean, std = float("nan"), float("nan")

            names.append(f"{name}")
            means.append(mean)
            stds.append(std)
            try:
                mean, std = benchmark(
                    dict(**args, context_length=context_length),
                    n_warmup=n_warmup,
                    n_step=n_step,
                    backward=False,
                    compile=enable_compile,
                    dtype=dtype,
                )
            except Exception as e:
                logger.error(f"Error benchmarking model {name} (forward only): {e}")
                mean, std = float("nan"), float("nan")

            names.append(f"{name} (forward only)")
            means.append(mean)
            stds.append(std)
        df = pd.DataFrame(
            {
                "model": names,
                "mean": means,
                "std": stds,
            }
        )
        return df.to_markdown()

    benchmark_sets = [(2, 10)]

    for n_warmup, n_step in benchmark_sets:
        result = run_benchmark(n_warmup, n_step)
        print(f"context length: {context_length}, {n_warmup} warmup, {n_step} step:")
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NOTE: recommend context_length: 128, 256, 512, 1024
    parser.add_argument("-l", "--context-length", type=int, default=256, help="Context length for the benchmark")
    parser.add_argument("-m", "--models", nargs="+", type=str, required=True, help="Model list for benchmarking")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model using torch.compile")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for the benchmark (float32 or float16 or bfloat16)",
    )
    args = parser.parse_args()
    logger.info(f"args: {vars(args)}")
    main(args)
