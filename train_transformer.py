from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from char_tokenizer import CharTokenizer
from transformer import TransformerLM, TransformerLMConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    block_size: int = 128
    batch_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    steps: int = 500
    lr: float = 3e-4
    temperature: float = 1.0
    seed: int = 42


def get_batches(token_ids: np.ndarray, block_size: int, batch_size: int, device: torch.device):
    x = torch.tensor(token_ids[:-1], dtype=torch.long, device=device)
    y = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    N = x.shape[0]
    while True:
        idx = torch.randint(0, N - block_size, (batch_size,), device=device)
        xs = torch.stack([x[i : i + block_size] for i in idx], dim=0)
        ys = torch.stack([y[i : i + block_size] for i in idx], dim=0)
        yield xs, ys


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        steps=args.steps,
        lr=args.lr,
        temperature=args.temperature,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    corpus_path = pathlib.Path("tiny_shakespeare.txt")
    if not corpus_path.exists():
        raise SystemExit("tiny_shakespeare.txt not found.")
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

    device = get_device()
    model_cfg = TransformerLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    model = TransformerLM(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    batches = get_batches(token_ids, cfg.block_size, cfg.batch_size, device)

    print("[Training Transformer LM]")
    print(f"Device: {device}")
    print(f"Vocab: {tokenizer.vocab_size}, Block: {cfg.block_size}, Layers: {cfg.n_layer}, Heads: {cfg.n_head}, Embd: {cfg.n_embd}")
    print("step\tloss")

    for step in range(1, cfg.steps + 1):
        x, y = next(batches)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(1, cfg.steps // 10) == 0 or step == 1:
            print(f"{step}\t{loss.item():.4f}")

        if step in {1, cfg.steps // 2, cfg.steps}:
            with torch.no_grad():
                start = torch.randint(0, len(token_ids) - cfg.block_size - 1, (1,), device=device)
                context = torch.tensor(token_ids[start : start + 1], dtype=torch.long, device=device)
                # seed with a single token id
                context = context.view(1, 1)
                out = model.generate(context, max_new_tokens=200, temperature=cfg.temperature)
                print("--- Sample ---")
                print(tokenizer.decode(out[0].tolist()))
                print()


if __name__ == "__main__":
    main()
