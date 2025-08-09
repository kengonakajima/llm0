from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore


def set_seed(seed: int) -> None:
    import random

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
class TrainLoopConfig:
    steps: int
    log_every: int = 50
    grad_clip: Optional[float] = None
    amp: bool = False
    accum_steps: int = 1
    sample_every: Optional[int] = None


# Types
Batch = Tuple[torch.Tensor, torch.Tensor]
SampleFn = Callable[[int], None]


def train(
    model: nn.Module,
    batch_iter: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    *,
    cfg: TrainLoopConfig,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    sample_fn: Optional[SampleFn] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Generic training loop.

    Expectations:
    - batch_iter yields (x, y) tensors already on `device` or will be moved here.
    - model forward signature: model(x, y) -> (logits, loss) with loss being a scalar tensor.
    - If sample_fn is provided, it's called as sample_fn(step) at configured intervals.
    """
    if device is None:
        device = get_device()

    model.train()

    scaler: Optional[torch.cuda.amp.GradScaler]
    if cfg.amp and (device.type == "cuda"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # progress iterator
    iterator = range(1, cfg.steps + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="train", ncols=0)

    optimizer.zero_grad(set_to_none=True)

    for step in iterator:  # type: ignore
        x, y = next(batch_iter)  # may already be on device
        if x.device != device:
            x = x.to(device)
        if y.device != device:
            y = y.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, loss = model(x, y)
            loss_to_backprop = loss / cfg.accum_steps
            scaler.scale(loss_to_backprop).backward()
        else:
            logits, loss = model(x, y)
            loss_to_backprop = loss / cfg.accum_steps
            loss_to_backprop.backward()

        should_step = (step % cfg.accum_steps) == 0
        if should_step:
            if cfg.grad_clip is not None:
                if scaler is not None:
                    # Unscale first, then clip
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        if (step % max(1, cfg.log_every) == 0) or (step == 1):
            msg = f"step={step} loss={loss.item():.4f}"
            if tqdm is not None:
                # Set postfix if using tqdm
                try:
                    iterator.set_postfix_str(msg)  # type: ignore
                except Exception:
                    pass
            else:
                print(msg)

        if sample_fn is not None and cfg.sample_every is not None:
            if (step % cfg.sample_every == 0) or (step == 1):
                with torch.no_grad():
                    sample_fn(step)
