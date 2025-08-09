from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple
import time

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
    # Early stopping (optional)
    early_stop_eval_interval: Optional[int] = None  # evaluate every N steps
    early_stop_min_steps: int = 0  # do not stop before this
    early_stop_patience: int = 0  # number of evals without improvement to tolerate
    early_stop_min_delta: float = 0.0  # relative improvement threshold (e.g., 0.005=0.5%)


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
    mem_cap_gb: Optional[float] = None,
    val_eval_fn: Optional[Callable[[], float]] = None,  # returns validation loss
) -> Tuple[float, float]:
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

    t0 = time.perf_counter()
    last_loss: Optional[float] = None

    best_val: Optional[float] = None
    stale_evals: int = 0

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

        # Track last observed loss (un-averaged over accum)
        last_loss = float(loss.item())

        # Optional memory cap (CUDA/MPS). Raises RuntimeError if exceeded.
        if mem_cap_gb is not None:
            cap_bytes = int(mem_cap_gb * (1024 ** 3))
            used_bytes: Optional[int] = None
            if device.type == "cuda":
                used_bytes = int(torch.cuda.memory_allocated(device))
            elif device.type == "mps":
                # Includes tensors allocated on MPS for this process
                used_bytes = int(torch.mps.current_allocated_memory())
            if used_bytes is not None and used_bytes > cap_bytes:
                raise RuntimeError(
                    f"Memory cap exceeded: {used_bytes / (1024**3):.2f}GB > {mem_cap_gb:.2f}GB on {device}"
                )

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

        # Early stopping check
        if cfg.early_stop_eval_interval and val_eval_fn is not None:
            if (step % cfg.early_stop_eval_interval == 0):
                current_val = float(val_eval_fn())
                if best_val is None:
                    best_val = current_val
                    stale_evals = 0
                else:
                    # relative improvement versus best
                    rel_improve = (best_val - current_val) / max(best_val, 1e-12)
                    if rel_improve >= cfg.early_stop_min_delta:
                        best_val = current_val
                        stale_evals = 0
                    else:
                        stale_evals += 1

                # Stop only after min_steps satisfied
                if (step >= max(1, cfg.early_stop_min_steps)) and (stale_evals > max(0, cfg.early_stop_patience)):
                    if tqdm is not None:
                        try:
                            iterator.close()  # type: ignore
                        except Exception:
                            pass
                    break

    elapsed_s = time.perf_counter() - t0
    return (last_loss if last_loss is not None else float("nan"), elapsed_s)
