from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from char_tokenizer import CharTokenizer
from training_loop import TrainLoopConfig, get_device, set_seed, train


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

    def forward(self, x_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """Return logits and optional loss to match common interface.

        x_ids: [B] -> logits: [B, V]
        targets: [B] or None
        """
        logits = self.logits_table[x_ids]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

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


def get_batches(token_ids: np.ndarray, batch_size: int, device: torch.device):
    """Yield random (x, y) pairs for bigram: shapes [B], [B]."""
    x_all = torch.tensor(token_ids[:-1], dtype=torch.long, device=device)
    y_all = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    N = x_all.shape[0]
    while True:
        idx = torch.randint(0, N, (batch_size,), device=device)
        yield x_all[idx], y_all[idx]


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

    device = get_device()
    model = TorchBigramLM(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)

    batches = get_batches(token_ids, cfg.batch_size, device)

    print("[Training Torch Bigram LM]")
    print(f"Device: {device}")
    print(f"Corpus length: {len(token_ids):,} tokens, Vocab: {tokenizer.vocab_size}")

    def sample_fn(step: int) -> None:
        start_id = int(torch.randint(0, tokenizer.vocab_size, (1,), device=device).item())
        out_ids = model.sample(length=200, start_id=start_id, temperature=cfg.temperature)
        print("\n--- Sample ---")
        print(tokenizer.decode(out_ids.tolist()))
        print()

    final_loss, elapsed_s = train(
        model,
        batches,
        optimizer,
        cfg=TrainLoopConfig(steps=cfg.steps, log_every=max(1, cfg.steps // 10), grad_clip=None, amp=False, accum_steps=1, sample_every=max(1, cfg.steps // 2)),
        scheduler=None,
        sample_fn=sample_fn,
        device=device,
        mem_cap_gb=16.0,
    )
    print(f"[Finished] steps={cfg.steps}, final_loss={final_loss:.4f}, elapsed={elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
