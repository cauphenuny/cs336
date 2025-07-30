from cs336_basics import Tokenizer, train_bpe
from utils import TempStringFile


def test_train_bpe_small():
    with TempStringFile("""
The quick brown fox jumps over the lazy dog. <|endoftext|>
test test test test test <|endoftext|>
    """) as f:
        vocab, merges = train_bpe(f.name, vocab_size=280, special_tokens=["<|endoftext|>"])
        # decoded_vocab = {k: v.decode("utf-8") for k, v in vocab.items()}
        # print(f"{vocab = }, {merges = }")


def test_train_bpe(file_path: str = "data/TinyStoriesV2-GPT4-train.txt", vocab_size: int = 10000):
    """
    Test the BPE training on a larger dataset.
    """
    tokenizer = Tokenizer.from_train(
        input_path=file_path, vocab_size=vocab_size, special_tokens=["<|endoftext|>"], num_processes=None
    )
    # print(f"{tokenizer.vocab = }, {tokenizer.merges = }")
    tokenizer.serialize(file_path.replace(".txt", f"-tokenizer-{vocab_size}.txt"))
    tokenizer.to_files(
        file_path.replace(".txt", f"-tokenizer-{vocab_size}-vocab.pkl"),
        file_path.replace(".txt", f"-tokenizer-{vocab_size}-merges.pkl"),
    )


if __name__ == "__main__":
    test_train_bpe_small()
    test_train_bpe(vocab_size=32000)
