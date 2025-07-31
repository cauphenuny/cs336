#!/Users/ycp/Source/Courses/cs336/assignments/assignment1-basics/.venv/bin/python
import cs336_basics
import random
import time
import sys
import argparse


def duration(func):
    start = time.time()
    ret = func()
    end = time.time()
    return (end - start, ret)


def test_tokenizer_simple():
    specials = ["<|endoftext|>"]
    tokenizer_10k = cs336_basics.Tokenizer.from_files(
        "data/TinyStoriesV2-GPT4-train-tokenizer-10000-vocab.json",
        "data/TinyStoriesV2-GPT4-train-tokenizer-10000-merges.json",
        specials,
    )
    test_string = "Hello, World!"
    encoded_ids = tokenizer_10k.encode(test_string)
    print(f"{tokenizer_10k.partial_decode(encoded_ids)}")
    decoded_string = tokenizer_10k.decode(encoded_ids)
    assert test_string == decoded_string


def test_tokenizer(test_filepath: str, tokenizer1_path: str, tokenizer2_path: str):
    specials = ["<|endoftext|>"]
    tokenizer_1 = cs336_basics.Tokenizer.from_files(
        f"{tokenizer1_path}-vocab.json",
        f"{tokenizer1_path}-merges.json",
        specials,
    )
    tokenizer_2 = cs336_basics.Tokenizer.from_files(
        f"{tokenizer2_path}-vocab.json",
        f"{tokenizer2_path}-merges.json",
        specials,
    )
    with open(test_filepath) as f:
        lines = f.readlines()

    sampled = random.sample(lines, 200000)
    print(f"Sampled lines: {len(sampled)}")
    joined_sample = "".join(sampled)
    print(f"Sampled length: {len(joined_sample)} characters")
    time_1, encoded_1 = duration(lambda: tokenizer_1.encode(joined_sample))
    time_2, encoded_2 = duration(lambda: tokenizer_2.encode(joined_sample))
    original_len = len("".join(sampled).encode("utf-8"))
    print(f"TinyStories ratio: {original_len / len(encoded_1)}")
    print(f"OpenWebText ratio: {original_len / len(encoded_2)}")
    print(f"TinyStories time: {time_1:.2f}s, {original_len / time_1:.2f} bytes/s")
    print(f"OpenWebText time: {time_2:.2f}s, {original_len / time_2:.2f} bytes/s")


if __name__ == "__main__":
    test_tokenizer_simple()
    # test_tokenizer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--t1", type=str, default="data/TinyStoriesV2-GPT4-train-tokenizer-10000")
    parser.add_argument("--t2", type=str, default="data/owt_valid-tokenizer-10000")
    args = parser.parse_args()
    test_tokenizer(args.file, args.t1, args.t2)
