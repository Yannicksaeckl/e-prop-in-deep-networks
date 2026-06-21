"""
Temporal XOR evidence accumulation task.

Each trial contains N cues (default N=7, odd).  Each cue occupies w=2 consecutive
timesteps:
  - Timestep 0 of cue: feature x0 ∈ {-1, +1} presented on channel 0.
  - Timestep 1 of cue: feature x1 ∈ {-1, +1} presented on channel 1.

Cue label  = XOR(sign x0, sign x1)  =  (x0 and x1 same sign) → 1 else 0
Trial label = majority vote of N cue labels.

After all cues there is a silent delay D, then a single decision step.

WHY DEPTH IS LOAD-BEARING
--------------------------
Computing the cue label requires combining information from two CONSECUTIVE steps:
x0 at step 0 must be held in memory until x1 arrives at step 1, then their product
sign evaluated.  A depth-1 RNN must use its single hidden layer for simultaneously
(a) buffering x0 across one step, (b) computing the XOR of x0 and x1, and
(c) maintaining the running evidence count across the full delay.  These three
demands compete for the same hidden capacity.

A depth-2 RNN can separate timescales: the bottom layer acts as a 1-step buffer
for x0 while the top layer maintains the long-running evidence accumulator.  The
cross-depth eligibility trace (eps_z / eps_cross) is exactly the gradient path
that carries credit from the top-layer accumulator back to the bottom-layer buffer.

INPUT CHANNELS  (N_IN = 5, fixed)
  0 : feature_0  — x0 ∈ {-1, +1} at step 0 of each cue, 0 elsewhere
  1 : feature_1  — x1 ∈ {-1, +1} at step 1 of each cue, 0 elsewhere
  2 : recall     — 1.0 at the decision step only
  3 : noise      — Gaussian (std = noise_level) at every step
  4 : bias       — constant 1.0

OUTPUT CHANNELS  (N_OUT = 2)
  0 : "left wins"
  1 : "right wins"

CONFIG KNOBS
  n_cues      : number of cues per trial (default 7; odd avoids majority ties)
  delay       : silent gap after last cue before decision (primary swept variable)
  w           : cue duration in steps; must be >= 2 (default 2)
  gap         : inter-cue silence in steps (default 2)
  noise_level : std of additive Gaussian noise on feature and noise channels
  seed        : per-call reproducibility seed (None = random)
  aux_running_readout : if True, adds low-weight auxiliary targets at each cue-end
                        step encoding the running evidence differential
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple

N_IN  = 5
N_OUT = 2
AUX_MASK_WEIGHT = 0.1


def xor_label(x0: Tensor, x1: Tensor) -> Tensor:
    """XOR of two binary features.

    x0, x1 : (...) tensors of binary features in {-1, +1}.
    Returns long tensor ∈ {0=left, 1=right} — 1 when x0 and x1 have the same sign.
    """
    return ((x0 * x1) > 0).long()


def generate_batch(
    batch_size: int,
    n_cues: int = 7,
    delay: int = 20,
    w: int = 2,
    gap: int = 2,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
    device: str = "cpu",
    aux_running_readout: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate a batch of temporal-XOR evidence accumulation trials.

    Returns
    -------
    inputs  : (T, B, N_IN)  float32
    targets : (T, B, 2)     float32 — one-hot at decision step
    mask    : (T, B)        float32 — 1.0 at decision step, 0 elsewhere
    """
    assert w >= 2, f"w must be >= 2 for temporal XOR (needs 2 feature steps), got {w}"

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    stride   = w + gap
    T        = n_cues * stride + delay + 1
    t_recall = T - 1
    B        = batch_size

    inputs  = torch.zeros(T, B, N_IN)
    targets = torch.zeros(T, B, N_OUT)
    mask    = torch.zeros(T, B)

    # Binary features for each trial × cue, ∈ {-1, +1}
    x0 = torch.randint(0, 2, (B, n_cues), generator=gen).float() * 2.0 - 1.0
    x1 = torch.randint(0, 2, (B, n_cues), generator=gen).float() * 2.0 - 1.0

    cue_labels = xor_label(x0, x1)   # (B, n_cues) ∈ {0, 1}

    # x0 at step 0, x1 at step 1 of each cue window
    for c in range(n_cues):
        t0 = c * stride
        t1 = c * stride + 1
        inputs[t0, :, 0] = x0[:, c] + torch.randn(B, generator=gen) * noise_level
        inputs[t1, :, 1] = x1[:, c] + torch.randn(B, generator=gen) * noise_level

    inputs[t_recall, :, 2] = 1.0                                          # recall
    inputs[:, :, 3] = torch.randn(T, B, generator=gen) * noise_level      # noise
    inputs[:, :, 4] = 1.0                                                  # bias

    # Majority vote
    right_count = (cue_labels == 1).sum(dim=1).float()
    left_count  = (cue_labels == 0).sum(dim=1).float()
    labels = (right_count > left_count).long()

    tied = right_count == left_count
    if tied.any():
        n_tied = int(tied.sum().item())
        labels[tied] = torch.randint(0, 2, (n_tied,), generator=gen)

    targets[t_recall, torch.arange(B), labels] = 1.0
    mask[t_recall] = 1.0

    if aux_running_readout:
        for c in range(n_cues):
            t_end     = c * stride + w - 1
            r_so_far  = (cue_labels[:, : c + 1] == 1).sum(1).float()
            l_so_far  = (cue_labels[:, : c + 1] == 0).sum(1).float()
            diff      = (r_so_far - l_so_far) / (c + 1)
            p_right   = torch.sigmoid(diff * 3.0)
            targets[t_end, :, 1] = p_right
            targets[t_end, :, 0] = 1.0 - p_right
            mask[t_end] = AUX_MASK_WEIGHT

    return inputs.to(device), targets.to(device), mask.to(device)


def sequence_length(n_cues: int = 7, delay: int = 20, w: int = 2, gap: int = 2) -> int:
    return n_cues * (w + gap) + delay + 1


def decision_timestep(n_cues: int = 7, delay: int = 20, w: int = 2, gap: int = 2) -> int:
    return sequence_length(n_cues, delay, w, gap) - 1


def n_in() -> int:
    return N_IN


def chance_accuracy() -> float:
    return 0.5


def task_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction correct at the decision step (mask > 0.5 steps only)."""
    decision = mask > 0.5
    pred     = logits.argmax(dim=-1)
    tgt      = targets.argmax(dim=-1)
    correct  = ((pred == tgt) * decision).sum().item()
    total    = decision.sum().item()
    return correct / total if total > 0 else 0.0
