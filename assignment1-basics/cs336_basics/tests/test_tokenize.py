import argparse
import os
import numpy as np
from loguru import logger
from cs336_basics.tokenize.tokenizer import Tokenizer
from cs336_basics.tokenize.pretokenizer import find_chunk_boundaries

def tokenize(tokenizer_path, file):
    specials = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        f"{tokenizer_path}-vocab.json",
        f"{tokenizer_path}-merges.json",
        specials
    )
    logger.info("reading")
    num_processes = min(os.cpu_count() or 1, 16)
    tokens: list[list[int]] = []
    with open(file, "rb") as f:
        logger.info(f"finding chunk boundaries, chunks = {num_processes}")
        boundaries = find_chunk_boundaries(
            f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            logger.info(f"reading chunk #{i}, size = {end - start}")
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            logger.info(f"tokenizing chunk #{i}")
            token = tokenizer.encode(chunk, verbose=True)
            logger.info(f"first 5 tokens: {tokenizer.partial_decode(token[:5])}")
            tokens.append(token)
    logger.info("concatenating token arrays")
    array = np.concatenate([np.array(t) for t in tokens])
    file_base = os.path.splitext(file)[0]
    logger.info("writing")
    np.save(f"{file_base}.npy", array)

def main():
    parser = argparse.ArgumentParser(description="Test Tokenization")
    parser.add_argument("tokenizer", type=str, help="Tokenizer to use")
    parser.add_argument("file", type=str, help="File to tokenize")
    args = parser.parse_args()
    print(f"Tokenizing file: {args.file}")
    tokenize(args.tokenizer, args.file)

if __name__ == "__main__":
    main()