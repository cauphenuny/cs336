import torch
import timeit
import statistics
import torch.cuda.nvtx as nvtx

from cs336_basics import network, optimize
from cs336_basics.network.multiplatform import ACCL_DEVICE, accl_module
from cs336_basics.network.models import TransformerModel


def benchmark(
    hyperparams: dict,
    vocab_size: int = 10000,
    batch_size: int = 4,
    n_warmup: int = 10,
    n_step: int = 100,
    backward: bool = True,
) -> tuple[float, float]:
    model = TransformerModel(**hyperparams, vocab_size=vocab_size).to(ACCL_DEVICE)
    optimizer = optimize.optimizers.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    len = model.context_length
    input = torch.randint(0, vocab_size, (batch_size, len)).to(ACCL_DEVICE)
    output = torch.randint(0, vocab_size, (batch_size, len)).to(ACCL_DEVICE)

    def run():
        with nvtx.range("forward pass"):
            logits = model(input)
        if backward:
            loss = network.functional.cross_entropy(logits, output).mean()
            with nvtx.range("backward pass"):
                loss.backward()
            with nvtx.range("optimize pass"):
                with nvtx.range("gradient clipping"):
                    optimize.functional.gradient_clip(model.parameters(), 2.0)
                with nvtx.range("optimize"):
                    optimizer.step()
        accl_module.synchronize()

    for _ in range(n_warmup):
        run()

    times = timeit.repeat(run, number=1, repeat=n_step)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n_step > 1 else 0.0

    return mean, std
