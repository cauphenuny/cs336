from cs336_basics import Tokenizer, train_bpe
from utils import TempStringFile
import os
import time
import argparse


def test_train_bpe_small():
    with TempStringFile("""
The quick brown fox jumps over the lazy dog. <|endoftext|>
test test test test test <|endoftext|>
    """) as f:
        vocab, merges = train_bpe(f.name, vocab_size=280, special_tokens=["<|endoftext|>"])
        # decoded_vocab = {k: v.decode("utf-8") for k, v in vocab.items()}
        print(f"{vocab = }, {merges = }")


def test_train_bpe(
    file_path: str = "data/TinyStoriesV2-GPT4-train.txt",
    vocab_size: int = 10000,
    num_processes: int | None = None,
    is_pretokenized: bool = False,
):
    """
    Test the BPE training on a larger dataset.
    """
    start = time.time()
    tokenizer = Tokenizer.from_train(
        input_path=file_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        num_processes=num_processes,
        is_pretokenized=is_pretokenized,
    )
    end = time.time()
    print(f"Tokenizer trained in {end - start:.2f} seconds")
    # print(f"{tokenizer.vocab = }, {tokenizer.merges = }")
    base, ext = os.path.splitext(file_path)
    tokenizer.serialize(f"{base}-tokenizer-{vocab_size}.txt")
    tokenizer.to_files(
        f"{base}-tokenizer-{vocab_size}-vocab.json",
        f"{base}-tokenizer-{vocab_size}-merges.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--pretokenized", type=bool, default=False, help="Whether the input file is pretokenized")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes to use for training BPE")
    args = parser.parse_args()

    test_train_bpe_small()
    test_train_bpe(
        file_path=args.file_path,
        vocab_size=args.vocab_size,
        num_processes=args.num_processes,
        is_pretokenized=args.pretokenized,
    )
