from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


def split_token_ids(
    token_ids: List[int],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> Tuple[List[int], List[int], List[int]]:
    """Split token id sequence into contiguous train/val/test.

    Ratios must sum to 1.0. No shuffling to preserve language continuity.
    """
    total = len(token_ids)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_ids = token_ids[:train_end]
    val_ids = token_ids[train_end:val_end]
    test_ids = token_ids[val_end:]
    return train_ids, val_ids, test_ids


@dataclass
class NextTokenBlockDataset:
    """Next-token prediction dataset over a single token sequence.

    For index i, returns (x, y) where:
    - x = token_ids[i : i + block_size]
    - y = token_ids[i + 1 : i + block_size + 1]
    """

    token_ids: List[int]
    block_size: int
    as_torch: bool = True

    def __post_init__(self) -> None:
        if self.as_torch and torch is None:
            # If torch is not available, silently fall back to Python lists
            self.as_torch = False
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if len(self.token_ids) <= self.block_size:
            raise ValueError("token sequence too short for given block_size")

    def __len__(self) -> int:
        # Number of (x,y) pairs we can form with given block size
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        end = index + self.block_size
        x_ids = self.token_ids[start:end]
        y_ids = self.token_ids[start + 1 : end + 1]
        if self.as_torch and torch is not None:
            return (
                torch.tensor(x_ids, dtype=torch.long),
                torch.tensor(y_ids, dtype=torch.long),
            )
        return x_ids, y_ids


def make_batch(dataset: NextTokenBlockDataset, batch_size: int, start_indices: Optional[List[int]] = None):
    """Create a simple batch of size B from given starting indices.

    If start_indices is None, uses evenly spaced indices.
    Returns tensors if dataset.as_torch else lists-of-lists.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    length = len(dataset)
    if start_indices is None:
        if batch_size == 1:
            start_indices = [0]
        else:
            step = max(1, length // (batch_size + 1))
            start_indices = [i * step for i in range(batch_size)]
    if len(start_indices) != batch_size:
        raise ValueError("start_indices length must match batch_size")

    xs = []
    ys = []
    for idx in start_indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)

    if dataset.as_torch and torch is not None:
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
    return xs, ys
