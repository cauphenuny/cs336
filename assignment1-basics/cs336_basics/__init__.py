import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from . import cpp_extensions  # type: ignore
from . import tokenize
from . import network
# from .network import layers
# from .network import functional
