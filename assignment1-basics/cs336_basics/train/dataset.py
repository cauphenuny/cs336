from jaxtyping import Int, Float
import numpy as np
import random
import torch
from typing import Type, Literal
from loguru import logger
import math
from typing import Optional


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
        self.data = np.load(path, mmap_mode="r")
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
        limit_type: Literal["total_tokens", "train_steps"] = "total_tokens",
        vocab_size: int | None = None,
        dataset_dtype=np.int16,
        device: torch.device | str | None = None,
    ):
        self.path = path
        self.data = np.load(path, mmap_mode="r")
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
        self.start_iter = 0
        logger.info(
            f"Loaded dataset {path}, data.dtype: {self.data.dtype}, data.shape: {self.data.shape}, context: {self.context_length}, batch: {self.batch_size}, max_iter: {self.max_iter}"
        )

    def __len__(self):
        return self.max_iter

    def __iter__(self):
        self.cur_iter = self.start_iter
        return self

    def __next__(self):
        if self.cur_iter < self.max_iter:
            self.cur_iter += 1
            return self._get_data()
        else:
            raise StopIteration

    def _get_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        input, target = get_batch(
            self.data,
            batch_size=self.batch_size,
            context_length=self.context_length,
            device=self.device,
        )
        if self.vocab_size:
            regenerate = False
            if torch.max(input) >= self.vocab_size:
                logger.warning(
                    f"Input token {torch.max(input)} exceeds vocab size {self.vocab_size}, regenerating batch"
                )
                regenerate = True
            if torch.max(target) >= self.vocab_size:
                logger.warning(
                    f"Target token {torch.max(target)} exceeds vocab size {self.vocab_size}, regenerating batch"
                )
                regenerate = True
            if regenerate:
                return self._get_data()
        return input, target

    def check(self):
        if not self.vocab_size:
            logger.warning("Vocab size not set, skipping check.")
            return
        for i, token in enumerate(self.data):
            if token >= self.vocab_size:
                logger.error(f"Token {token} at index {i} exceeds vocab size {self.vocab_size}, file: {self.path}")

    def set_start_iter(self, start_iter: int):
        self.start_iter = start_iter


class RandomWindowDataset(torch.utils.data.Dataset):
    """Map-style dataset exposing all possible (context_length) windows.
    __getitem__(i) returns (tokens[i : i+ctx], tokens[i+1 : i+ctx+1])
    This pairs with a RandomSampler(replacement=True) to mimic original get_batch randomness.
    """

    def __init__(self, path: str, context_length: int, vocab_size: int | None = None):
        self.path = path
        self.data = np.load(path, mmap_mode="r")  # memmap for large corpora
        self.context_length = context_length
        self.vocab_size = vocab_size
        logger.info(
            f"Loaded memmap dataset {path}, dtype={self.data.dtype}, shape={self.data.shape}, ctx={context_length}"
        )

    def __len__(self):
        return self.data.shape[0] - self.context_length

    def __getitem__(self, idx: int):
        x_np = self.data[idx : idx + self.context_length]
        y_np = self.data[idx + 1 : idx + self.context_length + 1]
        x = torch.from_numpy(np.asarray(x_np, dtype=np.int64))
        y = torch.from_numpy(np.asarray(y_np, dtype=np.int64))
        if self.vocab_size is not None:
            if x.max() >= self.vocab_size or y.max() >= self.vocab_size:
                # Fail fast; upstream preprocessing should ensure vocab bounds
                raise ValueError(
                    f"Token id exceeds vocab size {self.vocab_size} at window starting {idx}"
                )
        return x, y


class TorchTextDataLoader:
    """A DataLoader-based alternative to TextDataLoader, use native torch

    Arguments largely mirror TextDataLoader. Differences:
      - Uses torch DataLoader + RandomSampler(replacement=True) for random window sampling.
      - Supports num_workers, pin_memory for performance.
      - Returns (input, target) tensors shaped (batch_size, context_length) each iteration.
      - Device transfer (if provided) is done after batch collation (kept simple & explicit).
    """

    def __init__(
        self,
        path: str,
        context_length: int,
        batch_size: int,
        limit: int,
        limit_type: Literal["total_tokens", "train_steps"] = "total_tokens",
        vocab_size: int | None = None,
        device: torch.device | str | None = None,
        num_workers: int = 0,
        seed: int | None = 42,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
    ):
        self.path = path
        self.context_length = context_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.device = device

        # Compute max iterations consistent with legacy logic
        if limit_type == "train_steps":
            self.max_iter = limit
        elif limit_type == "total_tokens":
            self.max_iter = limit // batch_size // context_length
        else:
            raise ValueError(f"Unknown limit_type: {limit_type}")
        if self.max_iter <= 0:
            raise ValueError("Computed max_iter <= 0. Adjust limit / batch_size / context_length.")

        self.dataset = RandomWindowDataset(path, context_length, vocab_size=vocab_size)
        required_samples = self.max_iter * batch_size

        # Random sampler with replacement to mimic original random window sampling semantics
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        self.sampler = torch.utils.data.RandomSampler(
            self.dataset, replacement=True, num_samples=required_samples, generator=generator
        )

        # Build underlying DataLoader
        loader_kwargs: dict = dict(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        if num_workers > 0:
            # persistent_workers & prefetch_factor only valid when workers > 0
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
        self.loader = torch.utils.data.DataLoader(**loader_kwargs)

        self._iter = None
        self.cur_iter = 0
        self.start_iter = 0
        logger.info(
            f"Built TorchTextDataLoader path={path}, ctx={context_length}, batch={batch_size}, max_iter={self.max_iter}, workers={num_workers}"
        )

    def __len__(self):
        return self.max_iter

    def __iter__(self):
        self.cur_iter = self.start_iter
        self._iter = iter(self.loader)
        # Skip to start_iter if resuming
        for _ in range(self.start_iter):
            try:
                next(self._iter)
            except StopIteration:
                break
        return self

    def __next__(self):
        assert self._iter is not None
        if self.cur_iter >= self.max_iter:
            raise StopIteration
        try:
            batch = next(self._iter)
        except StopIteration:
            raise StopIteration
        self.cur_iter += 1
        x, y = batch  # both (B, ctx)
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y

    def set_start_iter(self, start_iter: int):
        if start_iter < 0 or start_iter >= self.max_iter:
            raise ValueError(f"start_iter {start_iter} out of range 0..{self.max_iter-1}")
        self.start_iter = start_iter

    # Optional compatibility method mirroring legacy check()
    def check(self):
        if not self.vocab_size:
            logger.warning("Vocab size not set, skipping check.")
            return
        # Sample a few indices to validate (avoid full scan for performance)
        for test_idx in [0, len(self.dataset)//4, len(self.dataset)//2, len(self.dataset)-1]:
            x, y = self.dataset[test_idx]
            if x.max() >= self.vocab_size or y.max() >= self.vocab_size:
                logger.error(
                    f"Token exceeds vocab size at window starting {test_idx} (max seen: {max(x.max().item(), y.max().item())})"
                )
                break
