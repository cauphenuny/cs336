import argparse
from cs336_basics.train.checkpoint import load_model
from cs336_basics.network.models import TransformerLM
from cs336_basics.tokenize.tokenizer import Tokenizer
from cs336_basics.optimize.optimizers import AdamW
from cs336_basics.train.decode import generate


def main(path, tokenizer_name, vocab_size=10000, context_length=256, temperature=1e-5, top_p=0.9):
    model = load_model(path)
    tokenizer = Tokenizer.from_name(tokenizer_name, vocab_size, special_tokens=["<|endoftext|>"])
    while True:
        prompt = input(">>> ")
        text = generate(
            prompt,
            model,
            tokenizer,
            max_length=context_length,
            temperature=temperature,
            top_p=top_p,
        )
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--temp", type=float, default=1e-5)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    main(args.model, args.tokenizer, temperature=args.temp, top_p=args.top_p)
