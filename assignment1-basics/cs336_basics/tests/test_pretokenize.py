import argparse
from cs336_basics.tokenize import pretokenizer
from utils import TempStringFile


def pretokenize():
    with TempStringFile("The quick brown fox jumps over the lazy dog. <|endoftext|>") as f:
        counts = pretokenizer.pretokenize_corpus(f.name, ["<|endoftext|>"])
        print(counts.most_common())


def file_pretokenize():
    word_counts = pretokenizer.pretokenize_corpus("data/TinyStoriesV2-GPT4-train.txt", ["<|endoftext|>"])
    print(word_counts.most_common(10))


def save_pretokenized(file_path: str, num_proc: int | None = None):
    word_counts = pretokenizer.pretokenize_corpus(file_path, special_tokens=["<|endoftext|>"], num_processes=num_proc)
    save_path = file_path.replace(".txt", "-pretokenized.pkl")
    pretokenizer.save(word_counts, save_path)
    print(f"Pretokenized data saved to {save_path}")


if __name__ == "__main__":
    pretokenize()
    # test_file_pretokenize()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/owt_valid.txt")
    parser.add_argument("--num_proc", type=int, default=0, help="Number of processes to use for pretokenization")
    args = parser.parse_args()
    save_pretokenized(args.file_path, args.num_proc if args.num_proc > 0 else None)
