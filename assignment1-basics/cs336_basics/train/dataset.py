from re import S
from click.core import F
from jaxtyping import Int, Float
import numpy as np
import random
import torch
from typing import Type
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

    input = np.array([x[i : i + context_length] for i in indices])
    target = np.array([x[i + 1 : i + context_length + 1] for i in indices])
    input = torch.tensor(input, device=device, dtype=torch.int64)
    target = torch.tensor(target, device=device, dtype=torch.int64)
    # logger.info(f"Batch: {input.shape = }, {target.shape = }, {input.dtype = }, {target.dtype = }")
    return input, target


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        context_length: int,
        device: torch.device | str | None = None,
        dataset_dtype=np.int16,
    ):
        self.path = path
        self.data = np.memmap(path, dtype=dataset_dtype, mode="r")
        # self.data = np.load(path)
        self.context_length = context_length
        self.device = device
        logger.info(
            f"Loaded dataset {path}, dtype: {self.data.dtype}, shape: {self.data.shape}, context: {self.context_length}"
        )

    def __len__(self):
        return self.data.shape[0] - self.context_length

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return torch.tensor(x, device=self.device, dtype=torch.int64), torch.tensor(
            y, device=self.device, dtype=torch.int64
        )


class TextDataLoader:
    def __init__(
        self,
        path: str,
        context_length: int,
        batch_size: int,
        limit: int,
        limit_type: str = "total_tokens",
        vocab_size: int | None = None,
        dataset_dtype=np.int16,
        device: torch.device | str | None = None,
    ):
        assert limit_type in ("total_tokens", "train_steps")
        self.data = np.memmap(path, dtype=dataset_dtype, mode="r")
        # self.data = np.load(path)
        self.context_length = context_length
        self.device = device
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        if limit_type == "train_steps":
            self.max_iter = limit
        elif limit_type == "total_tokens":
            self.max_iter = limit // self.batch_size // self.context_length
        else:
            raise ValueError(f"Unknown limit_type: {limit_type}")
        self.cur_iter = 0
        logger.info(
            f"Loaded dataset {path}, data.dtype: {self.data.dtype}, data.shape: {self.data.shape}, context: {self.context_length}, batch: {self.batch_size}, max_iter: {self.max_iter}"
        )

    def __len__(self):
        return self.max_iter

    def __iter__(self):
        self.cur_iter = 0
        return self

    def __next__(self):
        if self.cur_iter < self.max_iter:
            input, target = get_batch(
                self.data,
                batch_size=self.batch_size,
                context_length=self.context_length,
                device=self.device,
            )
            self.cur_iter += 1
            if self.vocab_size:
                assert torch.max(input) < self.vocab_size, (
                    f"Input token {torch.max(input)} exceeds vocab size {self.vocab_size}"
                )
                assert torch.max(target) < self.vocab_size, (
                    f"Target token {torch.max(target)} exceeds vocab size {self.vocab_size}"
                )
            return input, target
        else:
            raise StopIteration
