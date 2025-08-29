"""
Type stubs for cpp_extensions module.
"""

def hello() -> None:
    """A function that prints a hello message."""
    ...

def encode_bpe(
    input: list[tuple[bytes, ...]],
    merges: list[tuple[bytes, bytes]],
    vocab: dict[bytes, int],
    num_threads: int,
    verbose: bool = False,
) -> list[int]:
    """
    Encodes a list of byte tuples using BPE merges and a vocabulary.

    Args:
        input: A list of byte tuples to encode.
        merges: A list of BPE merge pairs.
        vocab: A dictionary mapping integer IDs to byte sequences.

    Returns:
        A list of integer IDs corresponding to the encoded tokens.
    """
    ...

def train_bpe(
    vocab: dict[int, bytes],
    word_counts: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: ...
