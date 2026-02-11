import importlib.metadata

__version__ = importlib.metadata.version("cs336-systems")

from .main import *
from . import cuda
