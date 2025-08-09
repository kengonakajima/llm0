from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import numpy as np
import torch

from char_tokenizer import CharTokenizer
from transformer import TransformerLM, TransformerLMConfig


def load_checkpoint(path: pathlib.Path, device: torch.device) -> tuple[TransformerLM, TransformerLMConfig]:
    data = torch.load(path, map_location=device)
    cfg_dict = data["config"]
    cfg = TransformerLMConfig(
        vocab_size=cfg_dict["vocab_size"],
        block_size=cfg_dict["block_size"],
        n_layer=cfg_dict["n_layer"],
        n_head=cfg_dict["n_head"],
        n_embd=cfg_dict["n_embd"],
    )
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(data["state_dict"]) 
    model.eval()
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="transformer_latest.pt")
    p.add_argument("--prompt", type=str, default="To be, or not to be:")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"))

    corpus_path = pathlib.Path("tiny_shakespeare.txt")
    if not corpus_path.exists():
        raise SystemExit("tiny_shakespeare.txt not found.")
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)

    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    model, cfg = load_checkpoint(ckpt_path, device)

    # Encode prompt (truncate to block_size-1 to allow at least one token to be generated)
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_ids = prompt_ids[-(cfg.block_size - 1) :]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(x, max_new_tokens=args.max_new, temperature=args.temperature)
    text_out = tokenizer.decode(out[0].tolist())
    print(text_out)


if __name__ == "__main__":
    main()
