from __future__ import annotations

import pathlib
from typing import List

from char_tokenizer import CharTokenizer
from dataset import NextTokenBlockDataset, split_token_ids, make_batch


def preview_dataset(block_size: int = 64, batch_size: int = 4) -> None:
    corpus_path = pathlib.Path("tiny_shakespeare.txt")
    if not corpus_path.exists():
        raise SystemExit("tiny_shakespeare.txt not found.")

    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    token_ids: List[int] = tokenizer.encode(text)

    train_ids, val_ids, test_ids = split_token_ids(token_ids)

    print("[Dataset Preview]")
    print(f"Total tokens: {len(token_ids):,}")
    print(f"Train/Val/Test: {len(train_ids):,} / {len(val_ids):,} / {len(test_ids):,}")
    print(f"Block size: {block_size}, Batch size: {batch_size}")

    # Build dataset on train split
    ds = NextTokenBlockDataset(train_ids, block_size=block_size, as_torch=False)
    print(f"Num training samples (len): {len(ds):,}")

    # Build a batch and show a few examples
    xs, ys = make_batch(ds, batch_size=batch_size)

    for b in range(batch_size):
        x_ids = xs[b]
        y_ids = ys[b]
        x_text = tokenizer.decode(x_ids)
        y_text = tokenizer.decode(y_ids)
        print(f"\n--- Sample {b+1} ---")
        print("x_ids[:20] =", x_ids[:20])
        print("y_ids[:20] =", y_ids[:20])
        print("x_text preview:")
        print(repr(x_text[:80]))
        print("y_text preview:")
        print(repr(y_text[:80]))


if __name__ == "__main__":
    preview_dataset()
