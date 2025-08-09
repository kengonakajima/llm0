from __future__ import annotations

import math
import pathlib
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from char_tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Prefer MPS on Apple Silicon if available
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    batch_size: int = 8192
    steps: int = 300
    lr: float = 3e-2
    temperature: float = 1.0
    seed: int = 42


class TorchBigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # logits for next-token given current token id: [V, V]
        self.logits_table = nn.Parameter(torch.zeros((vocab_size, vocab_size)))
        nn.init.normal_(self.logits_table, mean=0.0, std=0.01)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        # x_ids: [B] ints -> output logits [B, V]
        return self.logits_table[x_ids]

    @torch.no_grad()
    def sample(self, length: int, start_id: int, temperature: float = 1.0) -> np.ndarray:
        out = [start_id]
        for _ in range(length - 1):
            logits = self.logits_table[out[-1]]  # [V]
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            out.append(next_id)
        return np.array(out, dtype=np.int64)


def train_torch_bigram(tokenizer: CharTokenizer, token_ids: np.ndarray, cfg: TrainConfig) -> TorchBigramLM:
    print("[Training Torch Bigram LM]")
    device = get_device()
    V = tokenizer.vocab_size

    model = TorchBigramLM(V).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Build long tensors of pairs
    x_all = torch.tensor(token_ids[:-1], dtype=torch.long, device=device)
    y_all = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    N = x_all.shape[0]

    def eval_loss(sample_size: int = 20000) -> float:
        idx = torch.randint(0, N, (sample_size,), device=device)
        logits = model(x_all[idx])  # [B, V]
        loss = F.cross_entropy(logits, y_all[idx])
        return float(loss.item())

    print(f"Device: {device}")
    print(f"Total pairs: {N:,}, Vocab: {V}")
    print("step\tloss")

    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, N, (cfg.batch_size,), device=device)
        logits = model(x_all[idx])
        loss = F.cross_entropy(logits, y_all[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % max(1, cfg.steps // 10) == 0 or step == 1:
            print(f"{step}\t{loss.item():.4f}")

        if step in {1, cfg.steps // 2, cfg.steps}:
            with torch.no_grad():
                start_id = int(x_all[0].item())
                out_ids = model.sample(length=200, start_id=start_id, temperature=cfg.temperature)
                print("--- Sample ---")
                print(tokenizer.decode(out_ids.tolist()))
                print()

    return model


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, temperature=args.temperature, seed=args.seed)

    set_seed(cfg.seed)

    corpus_path = pathlib.Path("tiny_shakespeare.txt")
    if not corpus_path.exists():
        raise SystemExit("tiny_shakespeare.txt not found.")
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

    print("[Setup]")
    print(f"Corpus length: {len(token_ids):,} tokens, Vocab: {tokenizer.vocab_size}")

    _ = train_torch_bigram(tokenizer, token_ids, cfg)


if __name__ == "__main__":
    main()
