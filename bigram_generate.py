from __future__ import annotations

import pathlib
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from char_tokenizer import CharTokenizer


def build_bigram_counts(text: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    # Add start token handling by using \n as segment separator implicitly
    for a, b in zip(text, text[1:]):
        counts[(a, b)] += 1
    return counts


def to_transition_table(counts: Dict[Tuple[str, str], int]) -> Dict[str, List[Tuple[str, float]]]:
    outgoing: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (a, b), c in counts.items():
        outgoing[a][b] += c
    table: Dict[str, List[Tuple[str, float]]] = {}
    for a, next_counts in outgoing.items():
        total = sum(next_counts.values())
        # Sort next states by frequency desc for stable previews
        items = sorted(next_counts.items(), key=lambda kv: kv[1], reverse=True)
        table[a] = [(b, cnt / total) for b, cnt in items]
    return table


def sample_next(table: Dict[str, List[Tuple[str, float]]], prev_char: str, rng: random.Random) -> str:
    choices = table.get(prev_char)
    if not choices:
        # Fallback: choose any key if no outgoing edges
        any_state = rng.choice(list(table.keys()))
        choices = table[any_state]
    r = rng.random()
    acc = 0.0
    for ch, p in choices:
        acc += p
        if r <= acc:
            return ch
    return choices[-1][0]


def generate_text(table: Dict[str, List[Tuple[str, float]]], length: int, seed_char: str | None = None, seed: int = 42) -> str:
    rng = random.Random(seed)
    if seed_char is None:
        seed_char = rng.choice(list(table.keys()))
    out_chars = [seed_char]
    prev = seed_char
    for _ in range(length - 1):
        nxt = sample_next(table, prev, rng)
        out_chars.append(nxt)
        prev = nxt
    return "".join(out_chars)


def main() -> None:
    corpus_path = pathlib.Path("tiny_shakespeare.txt")
    if not corpus_path.exists():
        raise SystemExit("tiny_shakespeare.txt not found. Please place it in repo root.")
    text = corpus_path.read_text(encoding="utf-8")

    # Tokenizer is not strictly needed for bigram-by-char, but we show vocab size
    tokenizer = CharTokenizer.from_text(text)
    print("[Bigram Character Model]")
    print(f"Corpus length (chars): {len(text):,}")
    print(f"Vocab size (chars): {tokenizer.vocab_size}")

    counts = build_bigram_counts(text)
    # Show top-10 most frequent characters and bigrams
    from collections import Counter

    char_counts = Counter(text)
    top_chars = char_counts.most_common(10)
    bigram_counts = Counter(counts)
    top_bigrams = bigram_counts.most_common(10)

    print("Top-10 characters (char, count):", top_chars)
    print("Top-10 bigrams ((a,b), count):", top_bigrams)

    table = to_transition_table(counts)
    # Show a few transition rows
    sample_states = sorted(list(table.keys()))[:5]
    print("\nSample transition rows (char -> [(next, prob)...]):")
    for s in sample_states:
        print(s, "->", table[s][:5])

    print("\nGenerating samples (count-based):\n")

    for i in range(3):
        sample = generate_text(table, length=400, seed=42 + i)
        print(f"--- Sample {i+1} ---")
        print(sample)
        print()


if __name__ == "__main__":
    main()
