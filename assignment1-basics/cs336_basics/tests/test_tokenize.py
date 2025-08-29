import argparse
import os
import numpy as np
from loguru import logger
from cs336_basics.tokenize.tokenizer import Tokenizer

def tokenize(tokenizer_path, file):
    specials = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        f"{tokenizer_path}-vocab.json",
        f"{tokenizer_path}-merges.json",
        specials
    )
    logger.info("reading")
    with open(file, "r") as f:
        text = f.read()
        tokens = tokenizer.encode(text, verbose=True)
        first = tokenizer.partial_decode(tokens[:20])
        print(f"First 20 tokens: {tokens[:20]}, decoded: {first}")
        file_base, ext = os.path.splitext(file)
        np.save(f"{file_base}.npy", tokens)

def main():
    parser = argparse.ArgumentParser(description="Test Tokenization")
    parser.add_argument("tokenizer", type=str, help="Tokenizer to use")
    parser.add_argument("file", type=str, help="File to tokenize")
    args = parser.parse_args()
    print(f"Tokenizing file: {args.file}")
    tokenize(args.tokenizer, args.file)

if __name__ == "__main__":
    main()