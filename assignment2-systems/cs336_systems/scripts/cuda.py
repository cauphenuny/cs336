import cs336_systems
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

    def run_benchmark(n_warmup, n_step):
        names, means, stds = [], [], []
        for name, args in models.items():
            mean, std = cs336_systems.cuda.benchmark(
                dict(**args, context_length=context_length),
                n_warmup=n_warmup,
                n_step=n_step,
            )
            names.append(f"{name}")
            means.append(mean)
            stds.append(std)
            # mean, std = cs336_systems.cuda.benchmark(
            #     dict(**args, context_length=context_length),
            #     n_warmup=n_warmup,
            #     n_step=n_step,
            #     backward=False,
            # )
            # names.append(f"{name} (forward only)")
            # means.append(mean)
            # stds.append(std)
        df = pd.DataFrame(
            {
                "model": names,
                "mean": means,
                "std": stds,
            }
        )
        return df.to_markdown()

    benchmark_sets = [(5, 10), (0, 10), (1, 10), (2, 10)]

    for n_warmup, n_step in benchmark_sets:
        result = run_benchmark(n_warmup, n_step)
        print(f"context length: {context_length}, {n_warmup} warmup, {n_step} step:")
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NOTE: recommend context_length: 128, 256, 512, 1024
    parser.add_argument("-l", "--context-length", type=int, default=256, help="Context length for the benchmark")
    args = parser.parse_args()
    main(args)
