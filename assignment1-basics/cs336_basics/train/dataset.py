from jaxtyping import Int, Float
import numpy as np
import random
import torch
from loguru import logger


def get_batch(
    x: Int[np.ndarray, " dataset_len"],
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
    shuffle: bool = False,
):
    # logger.debug(f"Getting batch: {batch_size = }, {context_length = }, {x.shape = }")

    # def get_index(start_idx: int):
    #     if start_idx + context_length + 1 > x.shape[0]:
    #         start_idx = random.randint(0, x.shape[0] - context_length - 1)
    #     return start_idx

    # indices = [get_index(i) for i in range(batch_size)]
    # if shuffle:
    #     random.shuffle(indices)

    indices = torch.randint(0, x.shape[0] - context_length, (batch_size,))

    input = [x[i : i + context_length] for i in indices]
    target = [x[i + 1 : i + context_length + 1] for i in indices]
    input = torch.tensor(input, device=device)
    target = torch.tensor(target, device=device)
    return input, target
