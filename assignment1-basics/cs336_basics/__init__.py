import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from . import cpp_extensions  # type: ignore
from .tokenize import pretokenization
from .tokenize import tokenizer
from .tokenize.pretokenization import pretokenize_corpus, find_chunk_boundaries
from .tokenize.tokenizer import Tokenizer, train_bpe
from .model import layers
from .model import functional
