from cs336_basics.train.checkpoint import TransformerLM
import time
import argparse


def main(path, max_length=2048, temperature=1e-5, top_p=0.9):
    llm = TransformerLM.load(path)
    try:
        while True:
            prompt = input(">>> ")
            print(prompt, end="")
            start = time.time()
            for i, text in enumerate(
                llm.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    flush=True,
                )
            ):
                print(text, end="", flush=True)
            duration = time.time() - start
            print(f"\n(total: {i}, {i / duration:.3f}iter/s)")
    except KeyboardInterrupt:
        save = input("save model? [y/n]")
        if save.lower() == "y":
            path = input("Enter path to save model: ")
            name = input("Enter model name: ")
            llm.save(path, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temp", type=float, default=1e-5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    main(args.model, args.max_length, args.temp, args.top_p)
