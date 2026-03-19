import os
from loguru import logger

visible = os.environ.get("visible", None)
if visible is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["NPU_VISIBLE_DEVICES"] = visible
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible
    logger.info(f"Set visible devices to: {visible}")