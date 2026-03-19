try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    import os
    if os.environ.get("HCCL_HOST_SOCKET_PORT_RANGE", None) is None:
        os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"
    if os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", None) is None:
        os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"

except Exception as e:
    pass