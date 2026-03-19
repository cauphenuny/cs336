try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception as e:
    pass