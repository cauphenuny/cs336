import os

import argparse
import torch
from torch import nn
from torch import distributed as dist
from torch import multiprocessing as mp

from cs336_basics.distributed.ddp import DDP, DistributedSampler
from cs336_basics.optimize.optimizers import AdamW

torch.manual_seed(42)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.data = torch.randn(size, 10)
        self.labels = torch.cos(self.data.sum(dim=1)) + torch.randn(size) * 0.1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main(rank: int, world_size: int, batch_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
    )
    ddp_model = DDP(model)
    dataset = SimpleDataset(1000)
    sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size // world_size,
    )
    optimizer = AdamW(ddp_model.parameters(), lr=1e-3)

    for epoch in range(50):
        sampler.set_epoch(epoch)
        avg_loss = 0.0
        count = 0
        for batch in dataloader:
            data, labels = batch
            preds = ddp_model(data)
            loss = ((preds.sum(dim=1) - labels) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * data.size(0)
            count += data.size(0)
        avg_loss /= count

        # all-reduce loss
        avg_loss_tensor = torch.tensor(avg_loss)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size
        if rank == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=4, help="Number of processes to use for distributed training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    world_size = args.world_size
    batch_size = args.batch_size
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    mp.spawn(main, args=(world_size, batch_size), nprocs=world_size, join=True)
