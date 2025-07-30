from cs336_basics import pretokenize_corpus
from utils import TempStringFile


def test_pretokenize():
    with TempStringFile("The quick brown fox jumps over the lazy dog. <|endoftext|>") as f:
        counts = pretokenize_corpus(f.name, ["<|endoftext|>"])
        print(counts.most_common())


def test_file_pretokenize():
    word_counts = pretokenize_corpus("data/TinyStoriesV2-GPT4-train.txt", ["<|endoftext|>"])
    print(word_counts.most_common(10))


if __name__ == "__main__":
    test_pretokenize()
    test_file_pretokenize()
