from typing import TypeAlias
from collections import Counter
from collections.abc import Iterable
from loguru import logger
from tqdm import tqdm
from bidict import bidict
from .. import cpp_extensions
import os
import base64
import json
from . import pretokenizer


BytePair: TypeAlias = tuple[bytes, bytes]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    is_pretokenzied: bool = False,
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {i: s.encode("utf-8") for i, s in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    word_counts = Counter[tuple[bytes, ...]]()
    if is_pretokenzied:
        word_counts = pretokenizer.load(input_path)
    else:
        word_counts = pretokenizer.pretokenize_corpus(input_path, special_tokens, num_processes)

    pair_counts: Counter[BytePair] = Counter()
    for word, count in word_counts.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += count

    return cpp_extensions.train_bpe(vocab, dict(word_counts), dict(pair_counts), vocab_size)
    """
    with tqdm(total=vocab_size, desc="Training BPE") as pbar:
        while len(vocab) < vocab_size:
            if not pair_counts:
                break
            sorted_counts = sorted(pair_counts.items(), key=lambda e: (e[1], e[0]), reverse=True)
            best_pair, _best_count = sorted_counts[0]
            merges.append((best_pair[0], best_pair[1]))
            new_vocab = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = new_vocab

            while True:
                update_pair: list[tuple[tuple[BytePair, BytePair | None], int]] = []
                update_word: list[tuple[tuple[bytes, ...], tuple[bytes, ...]]] = []
                for word, count in word_counts.items():
                    for index in range(len(word) - 1):
                        if (word[index], word[index + 1]) == best_pair:
                            new_word = word[:index] + (new_vocab,) + word[index + 2 :]
                            # print(f"{word} ==> {new_word}, {count = }")
                            update_word.append((word, new_word))
                            update_pair.append((((best_pair), None), count))
                            if index > 0:
                                update_pair.append(
                                    (((word[index - 1], word[index]), (word[index - 1], new_vocab)), count)
                                )
                            if index < len(word) - 2:
                                update_pair.append(
                                    (((word[index + 1], word[index + 2]), (new_vocab, word[index + 2])), count)
                                )
                            break
                if not update_word:
                    break

                for word, new_word in update_word:
                    word_counts[new_word] += word_counts[word]
                    del word_counts[word]

                for (pair, new_pair), count in update_pair:
                    pair_counts[pair] -= count
                    assert pair_counts[pair] >= 0, f"Negative count for pair {pair}: {pair_counts[pair]}"
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                    if new_pair:
                        pair_counts[new_pair] += count

            pbar.n = len(vocab)
            pbar.refresh()

    return vocab, merges
    """


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.inverse_vocab = {b: i for i, b in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        for special in self.special_tokens:
            token = special.encode("utf-8")
            if token not in self.vocab.values():
                self.vocab[len(self.vocab)] = token

    @staticmethod
    def _merge(original: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]) -> tuple[bytes]:
        new_tokens = []
        i = 0
        while i < len(original):
            if i < len(original) - 1 and (original[i], original[i + 1]) == merge_pair:
                new_tokens.append(merge_pair[0] + merge_pair[1])
                i += 2
            else:
                new_tokens.append(original[i])
                i += 1
        return tuple(new_tokens)

    def encode(self, text: str, verbose: bool = False):
        if verbose:
            logger.info("pretokenizing")
        words = pretokenizer.pretokenize(text, self.special_tokens)
        if verbose:
            logger.info(f"pretokenized, {len(words)} words")
        # tokens: list[bytes] = []

        # for word in words:
        #     for merge_pair in self.merges:
        #         word = self._merge(word, merge_pair)
        #     for token in word:
        #         tokens.append(token)

        # return [self.vocab.inv[tok] for tok in tokens]
        num_threads = os.cpu_count() or 1
        if verbose:
            logger.info(f"tokenizing, {num_threads = }")
        return cpp_extensions.encode_bpe(words, self.merges, self.inverse_vocab, num_threads)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[token_id] for token_id in ids]
        byte_sequence = b"".join(tokens)
        return byte_sequence.decode("utf-8", errors="replace")

    def partial_decode(self, ids: list[int]) -> list[str]:
        return [self.vocab[token_id].decode("utf-8", errors="replace") for token_id in ids]

    def divide(self, text: str) -> list[str]:
        return self.partial_decode(self.encode(text))

    def serialize(self, output_path: str | os.PathLike) -> None:
        def serialize_token(b: bytes):
            try:
                return b.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                return repr(b)

        with open(output_path, "w", encoding="utf-8") as f:
            longest_token_length = max(len(token) for token in self.vocab.values())
            longest_tokens = [token for token in self.vocab.values() if len(token) == longest_token_length]
            f.write(f"{longest_token_length} {longest_tokens}\n")
            for i, token in self.vocab.items():
                f.write(f"{i} {serialize_token(token)}\n")
            for merge in self.merges:
                f.write(f"{serialize_token(merge[0])} {serialize_token(merge[1])}\n")

    @staticmethod
    def from_train(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        is_pretokenized: bool = False,
        num_processes: int | None = None,
    ) -> "Tokenizer":
        return Tokenizer(
            *train_bpe(input_path, vocab_size, special_tokens, is_pretokenized, num_processes), special_tokens
        )

    @staticmethod
    def from_files(
        vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        with open(vocab_filepath) as f:
            encoded_vocab = json.load(f)
            vocab = {int(k): base64.b64decode(v.encode("ascii")) for k, v in encoded_vocab.items()}
        with open(merges_filepath) as f:
            encoded_merges = json.load(f)
            merges = [
                (base64.b64decode(m[0].encode("ascii")), base64.b64decode(m[1].encode("ascii"))) for m in encoded_merges
            ]
        return Tokenizer(vocab, merges, special_tokens)  # type: ignore[return-value]

    def to_files(self, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike) -> None:
        encoded_vocab = {k: base64.b64encode(v).decode("ascii") for k, v in self.vocab.items()}
        encoded_merges = [
            (base64.b64encode(m[0]).decode("ascii"), base64.b64encode(m[1]).decode("ascii")) for m in self.merges
        ]

        with open(vocab_filepath, "w") as f:
            json.dump(encoded_vocab, f, ensure_ascii=True, indent=2)
        with open(merges_filepath, "w") as f:
            json.dump(encoded_merges, f, ensure_ascii=True, indent=2)
