import argparse
import torch
import timeit
import statistics
from loguru import logger

from cs336_basics.network import multiplatform
from cs336_basics.network.multiplatform import ACCL_DEVICE, accl_module
from cs336_basics.network.layers import MultiheadSelfAttention
from cs336_basics.network import functional as F


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value)


def main(args):
    batch_size = args.batch_size
    for d_model in [16, 32, 64, 128]:
        for seq_len in [256, 1024, 4096, 8192, 16384]:
            logger.info(f"Benchmarking d_model={d_model}, seq_len={seq_len}, batch_size={batch_size}")

            attn = Attention().to(ACCL_DEVICE)
            if args.compile:
                attn = torch.compile(attn, backend=multiplatform.compile_backend())
            result = torch.tensor([])

            def forward():
                nonlocal result
                query = torch.randn(batch_size, seq_len, d_model, device=ACCL_DEVICE, requires_grad=True)
                key = torch.randn(batch_size, seq_len, d_model, device=ACCL_DEVICE, requires_grad=True)
                value = torch.randn(batch_size, seq_len, d_model, device=ACCL_DEVICE, requires_grad=True)
                result = attn(query, key, value)
                accl_module.synchronize()

            def backward():
                result.mean().backward()
                accl_module.synchronize()

            start_memories = []
            end_memories = []
            forward_times = []
            backward_times = []

            for _ in range(args.num_warmup):
                forward()
                backward()

            for _ in range(args.num_steps):
                start_memories.append(multiplatform.device_memory_usage())
                forward_times.append(timeit.timeit(forward, number=1))
                end_memories.append(multiplatform.device_memory_usage())
                backward_times.append(timeit.timeit(backward, number=1))

            def stat(name, data):
                print(
                    f"{name}: mean={statistics.mean(data):.3f}, std={statistics.stdev(data) if len(data) > 1 else 0.0:.3f}"
                )

            stat("memory (start)", start_memories)
            stat("memory (end)", end_memories)
            stat("forward time", forward_times)
            stat("backward time", backward_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for the benchmark")
    parser.add_argument("-n", "--num_steps", type=int, default=100, help="Number of samples for the benchmark")
    parser.add_argument("--num_warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument(
        "--compile", action="store_true", help="Whether to use torch.compile to compile the attention module"
    )
    args = parser.parse_args()
    main(args)
