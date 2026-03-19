import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from . import prelude

from . import cpp_extensions  # type: ignore
from . import tokenize
from . import network
from . import optimize
from . import train
from . import distributed
# from .network import layers
# from .network import functional

__all__ = [
    "cpp_extensions",
    "tokenize",
    "network",
    "optimize",
    "train",
    "distributed",
    "prelude",
]