import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .tokenize.pretokenization import pretokenize_corpus, find_chunk_boundaries
from .tokenize.tokenizer import Tokenizer, train_bpe
from . import cpp_extensions  # type: ignore
