import argparse
import torch
from cs336_basics.network.models import TransformerModel
from cs336_basics.train.checkpoint import select_model, load_model
from cs336_basics.tokenize.tokenizer import Tokenizer


def main(tokenizer, vocab_size):
    tokenizer = Tokenizer.from_name(tokenizer, vocab_size=vocab_size)
    model = load_model(select_model())
    embed1 = model.embed(
        torch.tensor([tokenizer.token_id(" tree"), tokenizer.token_id(" trees")])
    )
    dist1 = embed1[0] - embed1[1]
    embed2 = model.embed(
        torch.tensor([tokenizer.token_id(" cat"), tokenizer.token_id(" cats")])
    )
    dist2 = embed2[0] - embed2[1]
    print(torch.cosine_similarity(dist1, dist2, dim=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()
    main(args.tokenizer, args.vocab_size)
