from loguru import logger
import torch
import os
from contextlib import contextmanager


def accl_type():
    if os.environ.get("ACCL"):
        return os.environ["ACCL"]

    try:
        import torch_npu  # pyright: ignore

        if torch_npu.npu.is_available():  # pyright: ignore
            return "npu"
    except ImportError:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


ACCL_TYPE = accl_type()


def accl_device(id: int | None = None) -> str:
    return f"{ACCL_TYPE}:{id}" if id is not None else ACCL_TYPE


ACCL_DEVICE = accl_device()


def get_accl_module():
    if ACCL_TYPE == "cuda":
        return torch.cuda
    if ACCL_TYPE == "mps":
        return torch.mps
    if ACCL_TYPE == "npu":
        import torch_npu  # pyright: ignore

        return torch_npu.npu  # pyright: ignore

    return torch.cpu


accl_module = get_accl_module()


def accl_activity():
    if ACCL_TYPE == "cuda":
        return torch.profiler.ProfilerActivity.CUDA
    elif ACCL_TYPE == "npu":
        import torch_npu  # pyright: ignore

        return torch_npu.profiler.ProfilerActivity.NPU
    return torch.profiler.ProfilerActivity.CPU


ACCL_ACTIVITY = accl_activity()


def compile_backend():
    if ACCL_TYPE == "mps":
        return "aot_eager"
    return "inductor"


@contextmanager
def profile(
    enable: bool = False,
    record_shapes: bool = True,
    with_stack: bool = False,
    profile_memory: bool = True,
    sort_bys: list[str] = ["cpu_time", "cpu_memory_usage"],
    row_limit: int = 30,
    json_trace_file: str | None = None,
):
    """
    A wrapper for torch.profiler.profile that:
      - Supports enable/disable (to avoid overhead when not profiling)
      - Detects backend (CUDA / MPS / CPU)
      - Falls back gracefully if GPU profiling is not supported (e.g., MPS)

    Args:
        enabled (bool): Whether to enable profiling
        record_shapes (bool): Whether to record tensor shapes
        with_stack (bool): Whether to record Python stack traces
        profile_memory (bool): Whether to track memory usage
        sort_by (str): Column to sort profiling results
        row_limit (int): Max rows in profiling table
        json_trace_file (str): Path to write JSON trace (if not None)
    """
    if not enable:
        # No-op context manager
        yield None
        return

    # Detect activities
    activities = [torch.profiler.ProfilerActivity.CPU]

    if ACCL_ACTIVITY != torch.profiler.ProfilerActivity.CPU:
        activities.append(ACCL_ACTIVITY)

    logger.info("Profiling with activities: {}", activities)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=3, active=10, repeat=3),
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=profile_memory,
        on_trace_ready=(torch.profiler.tensorboard_trace_handler(json_trace_file) if json_trace_file else None),
    ) as prof:
        yield prof
        for sort_by in sort_bys:
            logger.info("Profiling results sorted by {}", sort_by)
            print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))


def device_memory_usage():
    if ACCL_TYPE == "cuda":
        return torch.cuda.memory_allocated()
    elif ACCL_TYPE == "mps":
        return torch.mps.current_allocated_memory()
    else:
        raise NotImplementedError("device_memory_usage is only implemented for CUDA and MPS backends")
