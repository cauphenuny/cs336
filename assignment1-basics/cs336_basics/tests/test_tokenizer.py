import cs336_basics
import random
import time


def duration(func):
    start = time.time()
    ret = func()
    end = time.time()
    return (end - start, ret)


def test_tokenizer(test_filepath: str = "data/TinyStoriesV2-GPT4-valid.txt"):
    specials = ["<|endoftext|>"]
    tokenizer_10k = cs336_basics.Tokenizer.from_files(
        "data/TinyStoriesV2-GPT4-train-tokenizer-10000-vocab.pkl",
        "data/TinyStoriesV2-GPT4-train-tokenizer-10000-merges.pkl",
        specials,
    )
    tokenizer_32k = cs336_basics.Tokenizer.from_files(
        "data/TinyStoriesV2-GPT4-train-tokenizer-32000-vocab.pkl",
        "data/TinyStoriesV2-GPT4-train-tokenizer-32000-merges.pkl",
        specials,
    )
    with open(test_filepath) as f:
        lines = f.readlines()

    sampled = random.sample(lines, 50)
    print(f"Sampled lines: {sampled}")
    time_10k, encoded_10k = duration(lambda: tokenizer_10k.encode("".join(sampled)))
    time_32k, encoded_32k = duration(lambda: tokenizer_32k.encode("".join(sampled)))
    original_len = len("".join(sampled).encode("utf-8"))
    print(f"10k ratio: {original_len / len(encoded_10k)}")
    print(f"32k ratio: {original_len / len(encoded_32k)}")
    print(f"10k time: {time_10k:.2f}s, {original_len / time_10k:.2f} bytes/s")
    print(f"32k time: {time_32k:.2f}s, {original_len / time_32k:.2f} bytes/s")


if __name__ == "__main__":
    test_tokenizer()
    # test_tokenizer("data/owt_valid.txt")
