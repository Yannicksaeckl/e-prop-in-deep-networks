"""One-bit store-and-recall task for sparse temporal credit assignment.

Inputs have three channels at every timestep:
  0. value  -- a binary distractor/value bit
  1. store  -- one at the store command timestep
  2. recall -- one at the recall command timestep

The target is a binary class label, used only where ``recall_mask`` is true.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def _generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def generate_batch(
    batch_size: int,
    delay: int = 20,
    *,
    t_store: int = 1,
    seed: Optional[int] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(inputs, targets, recall_mask)`` for the 1-bit task.

    Parameters
    ----------
    batch_size:
        Number of independent trials.
    delay:
        Number of timesteps between store and recall.
    t_store:
        Store command timestep. The default leaves one distractor timestep
        before the store command.
    seed:
        Optional local seed. Passing it makes batch generation deterministic
        without changing global PyTorch RNG state.

    Shapes
    ------
    inputs:
        ``(T, B, 3)`` float tensor with ``[value, store, recall]`` channels.
    targets:
        ``(T, B)`` long tensor. Only the recall timestep is meaningful.
    recall_mask:
        ``(T, B)`` bool tensor, true at exactly one timestep per trial.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if delay <= 0:
        raise ValueError("delay must be positive")
    if t_store < 0:
        raise ValueError("t_store must be non-negative")

    dev = torch.device(device)
    gen = _generator(dev, seed)
    t_recall = t_store + delay
    T = t_recall + 1

    value_bits = torch.randint(
        0,
        2,
        (T, batch_size),
        generator=gen,
        device=dev,
        dtype=torch.long,
    )
    stored_bits = value_bits[t_store].clone()

    inputs = torch.zeros(T, batch_size, 3, device=dev, dtype=dtype)
    inputs[:, :, 0] = value_bits.to(dtype=dtype)
    inputs[t_store, :, 1] = 1.0
    inputs[t_recall, :, 2] = 1.0

    targets = torch.zeros(T, batch_size, device=dev, dtype=torch.long)
    targets[t_recall] = stored_bits

    recall_mask = torch.zeros(T, batch_size, device=dev, dtype=torch.bool)
    recall_mask[t_recall] = True

    return inputs, targets, recall_mask


def masked_cross_entropy(logits: Tensor, targets: Tensor, recall_mask: Tensor) -> Tensor:
    """Cross-entropy over the masked recall timestep(s)."""
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape (T, B, 2)")
    selected_logits = logits[recall_mask]
    selected_targets = targets[recall_mask]
    if selected_logits.numel() == 0:
        raise ValueError("recall_mask selects no timesteps")
    return torch.nn.functional.cross_entropy(selected_logits, selected_targets)


def task_accuracy(logits: Tensor, targets: Tensor, recall_mask: Tensor) -> float:
    """Recall accuracy over masked timesteps."""
    pred = logits.argmax(dim=-1)
    correct = (pred[recall_mask] == targets[recall_mask]).sum().item()
    total = int(recall_mask.sum().item())
    return correct / total if total else 0.0
