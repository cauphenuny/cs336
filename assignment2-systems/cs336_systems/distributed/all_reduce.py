import os
import multiprocessing as mp
import torch
import pandas as pd
from timeit import timeit

settings = [
    ("cpu", "gloo"),
    ("cuda", "nccl"),
]
sizes = [
    2**20,  # 1M
    10 * 2**20,  # 10M
    100 * 2**20,  # 100M
    2**30,  # 1B
]
nprocs = [2, 4, 6]
warmup = 5
benchmark_runs = 10

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

def benchmark_worker(
    rank: int,
    world_size: int,
    device: str,
    backend: str,
    size: int,
    warmup_runs: int,
    timed_runs: int,
    result_queue,
):
    setup(rank, world_size, backend)

    if device == "cuda":
        torch.cuda.set_device(rank)
        tensor_device = torch.device(f"cuda:{rank}")
    else:
        tensor_device = torch.device("cpu")

    tensor = torch.randn(size, device=tensor_device)

    def collective_step():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        if device == "cuda":
            torch.cuda.synchronize(rank)

    for _ in range(warmup_runs):
        collective_step()

    elapsed = timeit(collective_step, number=timed_runs) / timed_runs
    if rank == 0:
        result_queue.put(elapsed)

    torch.distributed.destroy_process_group()

def benchmark(setting: tuple[str, str], size: int, nproc: int) -> float:
    device, backend = setting
    spawn_ctx = mp.get_context("spawn")
    result_queue = spawn_ctx.SimpleQueue()
    torch.multiprocessing.spawn(
        benchmark_worker,
        args=(nproc, device, backend, size, warmup, benchmark_runs, result_queue),
        nprocs=nproc,
        join=True,
    )
    return result_queue.get()

def is_valid_config(setting: tuple[str, str], nproc: int) -> tuple[bool, str]:
    device, _ = setting
    if device != "cuda":
        return True, ""
    if not torch.cuda.is_available():
        return False, "cuda unavailable"
    cuda_count = torch.cuda.device_count()
    if nproc > cuda_count:
        return False, f"nproc({nproc}) > cuda_count({cuda_count})"
    return True, ""

def main():
    rows: list[dict[str, str | int | float | None]] = []
    for setting in settings:
        for size in sizes:
            for nproc in nprocs:
                device, backend = setting
                base_row = {
                    "device": device,
                    "backend": backend,
                    "size": size,
                    "nproc": nproc,
                    "warmup": warmup,
                    "benchmark_runs": benchmark_runs,
                }
                valid, reason = is_valid_config(setting, nproc)
                if not valid:
                    rows.append({**base_row, "avg_time_s": None, "status": f"skipped: {reason}"})
                    continue

                try:
                    avg_time = benchmark(setting, size, nproc)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    rows.append({**base_row, "avg_time_s": None, "status": f"error: {type(exc).__name__}"})
                    continue

                rows.append({**base_row, "avg_time_s": avg_time, "status": "ok"})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

"""
result

device backend       size  nproc  warmup  benchmark_runs  avg_time_s status
   cpu    gloo    1048576      2       5              10    0.001509     ok
   cpu    gloo    1048576      4       5              10    0.003057     ok
   cpu    gloo    1048576      6       5              10    0.006234     ok
   cpu    gloo   10485760      2       5              10    0.015649     ok
   cpu    gloo   10485760      4       5              10    0.049406     ok
   cpu    gloo   10485760      6       5              10    0.054533     ok
   cpu    gloo  104857600      2       5              10    0.234054     ok
   cpu    gloo  104857600      4       5              10    0.534037     ok
   cpu    gloo  104857600      6       5              10    0.589058     ok
   cpu    gloo 1073741824      2       5              10    1.887960     ok
   cpu    gloo 1073741824      4       5              10    4.685661     ok
   cpu    gloo 1073741824      6       5              10    5.488045     ok
  cuda    nccl    1048576      2       5              10    0.000116     ok
  cuda    nccl    1048576      4       5              10    0.000192     ok
  cuda    nccl    1048576      6       5              10    0.000242     ok
  cuda    nccl   10485760      2       5              10    0.000609     ok
  cuda    nccl   10485760      4       5              10    0.000551     ok
  cuda    nccl   10485760      6       5              10    0.000645     ok
  cuda    nccl  104857600      2       5              10    0.003485     ok
  cuda    nccl  104857600      4       5              10    0.004501     ok
  cuda    nccl  104857600      6       5              10    0.005123     ok
  cuda    nccl 1073741824      2       5              10    0.029820     ok
  cuda    nccl 1073741824      4       5              10    0.041525     ok
  cuda    nccl 1073741824      6       5              10    0.045988     ok
"""