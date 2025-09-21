import torch
import timeit
import statistics

from cs336_basics.network.multiplatform import ACCL_DEVICE, accl_module
from cs336_basics.network.models import TransformerModel
from cs336_basics.network.functional import cross_entropy


def benchmark(
    hyperparams: dict,
    vocab_size: int = 10000,
    batch_size: int = 4,
    n_warmup: int = 10,
    n_step: int = 100,
    backward: bool = True,
) -> tuple[float, float]:
    model = TransformerModel(**hyperparams, vocab_size=vocab_size).to(ACCL_DEVICE)
    len = model.context_length
    input = torch.randint(0, vocab_size, (batch_size, len)).to(ACCL_DEVICE)
    output = torch.randint(0, vocab_size, (batch_size, len)).to(ACCL_DEVICE)

    def run():
        logits = model(input)
        if backward:
            loss = cross_entropy(logits, output).mean()
            loss.backward()
        accl_module.synchronize()

    for _ in range(n_warmup):
        run()

    times = timeit.repeat(run, number=1, repeat=n_step)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n_step > 1 else 0.0

    return mean, std
