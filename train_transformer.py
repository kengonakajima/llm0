from __future__ import annotations

import pathlib
from dataclasses import dataclass
import math

import numpy as np
import torch

from char_tokenizer import CharTokenizer
from dataset import split_token_ids
from transformer import TransformerLM, TransformerLMConfig
from training_loop import TrainLoopConfig, get_device, set_seed, train


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


@torch.no_grad()
def eval_loss_and_ppl(model: torch.nn.Module, token_ids: np.ndarray, block_size: int, device: torch.device, *, batches: int = 20, batch_size: int = 128) -> tuple[float, float]:
    model.eval()
    x = torch.tensor(token_ids[:-1], dtype=torch.long, device=device)
    y = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    N = x.shape[0]
    total_loss = 0.0
    for _ in range(batches):
        idx = torch.randint(0, N - block_size, (batch_size,), device=device)
        xs = torch.stack([x[i : i + block_size] for i in idx], dim=0)
        ys = torch.stack([y[i : i + block_size] for i in idx], dim=0)
        _, loss = model(xs, ys)
        total_loss += float(loss.item())
    avg_loss = total_loss / max(1, batches)
    ppl = float(np.exp(avg_loss))
    model.train()
    return avg_loss, ppl


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
    train_ids, val_ids, test_ids = split_token_ids(list(token_ids))
    train_ids = np.array(train_ids, dtype=np.int64)
    val_ids = np.array(val_ids, dtype=np.int64)

    device = get_device()
    model_cfg = TransformerLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    model = TransformerLM(model_cfg).to(device)

    # AdamW with decoupled weight decay and parameter groups (exclude bias and norm params)
    no_decay_keys = ["bias", "ln", "norm"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name.lower() for k in no_decay_keys):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)

    # Warmup + Cosine decay scheduler over total steps
    total_steps = max(1, cfg.steps)
    warmup_steps = max(1, int(total_steps * 0.05))
    def lr_lambda(step_idx: int):
        # step_idx starts at 0 in LambdaLR
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, (total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    batches = get_batches(train_ids, cfg.block_size, cfg.batch_size, device)

    print("[Training Transformer LM]")
    print(f"Device: {device}")
    print(f"Vocab: {tokenizer.vocab_size}, Block: {cfg.block_size}, Layers: {cfg.n_layer}, Heads: {cfg.n_head}, Embd: {cfg.n_embd}")

    def sample_fn(step: int) -> None:
        # Validation metrics (for visibility)
        val_loss, val_ppl = eval_loss_and_ppl(model, val_ids, cfg.block_size, device, batches=10, batch_size=min(256, cfg.batch_size))
        print(f"\n[Val] step={step} loss={val_loss:.4f} ppl={val_ppl:.2f}")
        with torch.no_grad():
            # Seed with a single random token id
            start_id = int(torch.randint(0, tokenizer.vocab_size, (1,), device=device).item())
            context = torch.tensor([[start_id]], dtype=torch.long, device=device)
            out = model.generate(context, max_new_tokens=200, temperature=cfg.temperature)
            print("\n--- Sample ---")
            print(tokenizer.decode(out[0].tolist()))
            print()

    # Early stopping evaluator: returns validation loss only
    def val_eval_fn() -> float:
        val_loss, _ = eval_loss_and_ppl(model, val_ids, cfg.block_size, device, batches=10, batch_size=min(256, cfg.batch_size))
        return val_loss

    final_loss, elapsed_s = train(
        model,
        batches,
        optimizer,
        cfg=TrainLoopConfig(
            steps=cfg.steps,
            log_every=max(1, cfg.steps // 10),
            grad_clip=1.0,
            amp=False,
            accum_steps=1,
            sample_every=max(1, cfg.steps // 2),
            early_stop_eval_interval=200,
            early_stop_min_steps=1000,
            early_stop_patience=5,
            early_stop_min_delta=0.005,
        ),
        scheduler=scheduler,
        sample_fn=sample_fn,
        device=device,
        mem_cap_gb=16.0,
        val_eval_fn=val_eval_fn,
    )
    # Final validation after training
    val_loss, val_ppl = eval_loss_and_ppl(model, val_ids, cfg.block_size, device, batches=20, batch_size=min(256, cfg.batch_size))
    print(f"[Finished] steps={cfg.steps}, final_loss={final_loss:.4f}, elapsed={elapsed_s:.2f}s, val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")


if __name__ == "__main__":
    main()
