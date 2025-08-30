import argparse
import os
import numpy as np
from loguru import logger
from cs336_basics.tokenize.tokenizer import Tokenizer
from cs336_basics.tokenize.pretokenizer import find_chunk_boundaries

def tokenize(tokenizer_path, file, num_chunks=0, num_threads=0, dtype=np.int16):
    specials = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        f"{tokenizer_path}-vocab.json",
        f"{tokenizer_path}-merges.json",
        specials
    )
    logger.info("reading")
    num_chunks = num_chunks or min(os.cpu_count() or 1, 16)
    # tokens: list[list[int]] = []
    file_base = os.path.splitext(file)[0]
    with open(file, "rb") as f:
        logger.info(f"finding chunk boundaries, chunks = {num_chunks}")
        boundaries = find_chunk_boundaries(
            f, num_chunks, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            logger.info(f"reading chunk #{i}, size = {end - start}")
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            logger.info(f"tokenizing chunk #{i}")
            token = tokenizer.encode(chunk, verbose=True, num_threads=num_threads)
            logger.info(f"first 5 tokens: {tokenizer.partial_decode(token[:5])}")
            # tokens.append(token)
            array = np.array(token, dtype=dtype)
            logger.info(f"writing chunk #{i}, size = {array.nbytes / 1024 / 1024:,.2f} MB")
            np.save(f"{file_base}_chunk{i}.npy", array)
    logger.info("concatenating")
    arrays = [np.load(f"{file_base}_chunk{i}.npy") for i in range(len(boundaries) - 1)]
    final = np.concatenate(arrays)
    logger.info(f"writing final, size = {final.nbytes / 1024 / 1024:,.2f} MB")
    np.save(f"{file_base}.npy", final)
    for i in range(len(boundaries) - 1):
        os.remove(f"{file_base}_chunk{i}.npy")

def main():
    parser = argparse.ArgumentParser(description="Test Tokenization")
    parser.add_argument("tokenizer", type=str, help="Tokenizer to use")
    parser.add_argument("file", type=str, help="File to tokenize")
    parser.add_argument("-t", type=int, default=0, help="threads")
    parser.add_argument("-c", type=int, default=0, help="chunks")
    args = parser.parse_args()
    print(f"Tokenizing file: {args.file}")
    tokenize(args.tokenizer, args.file, num_chunks=args.c, num_threads=args.t)

if __name__ == "__main__":
    main()