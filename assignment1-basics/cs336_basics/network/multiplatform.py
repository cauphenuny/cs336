import torch
import os


def accl_type():
    if os.environ.get("ACCL"):
        return os.environ["ACCL"]

    try:
        import torch_npu

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
        import torch_npu

        return torch_npu.npu  # pyright: ignore

    return torch.cpu


accl_module = get_accl_module()
