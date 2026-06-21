"""
Enriched evidence accumulation task for testing depth-loading in deep e-prop.

Each trial presents N cues (default N=7, odd), each a d_cue-dimensional pattern
shown for w steps with a gap between cues, followed by a silent delay D and a
single decision step.  The network must report the majority side (left/right).

DEPTH-LOADING DESIGN
--------------------
Cue label is a degree-2 boolean (parity tree) function of the cue pattern:
  Step 1 — compute pairwise products:  z[j] = x[2j] * x[2j+1]   (j=0..m-1)
  Step 2 — compute weighted sum:       label = sign(Σ_j w[j] * z[j])

where x ∈ {-1,+1}^k are the k "relevant" binary coordinates of the cue pattern,
m = k/2, and w ∈ {-1,+1}^m are fixed random weights (generated from weights_seed).

The remaining d_cue - k coordinates are pure Gaussian noise (distractors).

Why depth is load-bearing
  A depth-2 tanh stack can factorize: layer-1 computes the pairwise products z[j],
  layer-2 accumulates the signed evidence.  A depth-1 stack must do both in one
  tanh stage, which is harder when k is large and n_rec is limited.
  The depth screen (Gate 1) verifies this empirically before the main experiment.

Input channels (n_in = d_cue + 3)
  0 .. d_cue-1 : cue pattern — active during each cue window, 0 during gaps/delay
  d_cue        : recall signal — 1.0 at the decision step only
  d_cue+1      : Gaussian noise (std = noise_level, every step)
  d_cue+2      : bias (constant 1.0)

Output channels (n_out = 2)
  0 : "left wins"   one-hot at decision step
  1 : "right wins"

Config knobs (all exposed as generate_batch arguments)
  n_cues        : number of cues per trial (default 7; odd avoids majority ties)
  d_cue         : cue pattern dimension (default 16)
  k             : number of relevant binary features (default 8; must be even)
  w             : cue duration in steps (default 1)
  gap           : inter-cue silence in steps (default 5)
  delay         : silent gap between last cue and decision (primary swept variable)
  noise_level   : std of Gaussian noise added to cue channels (default 0.1)
  weights_seed  : seed for generating fixed parity weights (default 42)
  seed          : per-call reproducibility seed (None = random)

aux_running_readout (default False)
  If True, adds a low-weight (0.1) auxiliary MSE target at each cue-end step,
  reading the running left/right count differential.  This raises early-timestep
  cosine SNR but dilutes the temporal-credit stress that the main experiment
  measures.  Keep False for headline runs; the option exists for ablations.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple

N_OUT = 2  # 0=left wins, 1=right wins
AUX_MASK_WEIGHT = 0.1   # low-weight auxiliary steps (when aux_running_readout=True)


# ── Parity-tree label function ────────────────────────────────────────────────

def make_parity_weights(k: int, seed: int = 42) -> Tensor:
    """Fixed continuous weights for the degree-2 boolean label function.

    There are m = k//2 pairwise sub-features z[j] = x[2j] * x[2j+1].
    Label = sign(Σ_j w[j]*z[j]) → {0=left, 1=right}.

    Weights are standard-normal (continuous, not binary ±1) so that ties
    (s=0) have measure zero and the label distribution is exactly balanced:
    P(label=0) = P(label=1) = 0.5 for any generic weight vector, since the
    distribution of s = Σ w[j]*z[j] is symmetric around 0 when z[j] are i.i.d.
    Rademacher.  Binary ±1 weights would create a 3/4 imbalance due to ties.

    Parameters
    ----------
    k    : number of relevant binary features (must be even)
    seed : seed for weight generation — fix this for a consistent task

    Returns
    -------
    weights : (m,) float32 tensor  where  m = k//2
    """
    assert k % 2 == 0, f"k must be even, got {k}"
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(k // 2).astype(np.float32)
    return torch.tensor(w)


def parity_label(x: Tensor, parity_weights: Tensor) -> Tensor:
    """Compute per-trial cue labels via the degree-2 parity tree.

    Parameters
    ----------
    x              : (B, k) binary features ∈ {-1, +1}
    parity_weights : (m,) fixed ±1 weights;  m = k//2

    Returns
    -------
    labels : (B,) long tensor ∈ {0=left, 1=right}
    """
    m = parity_weights.shape[0]
    # Pairwise products — shape (B, m)
    z = x[:, :2 * m:2] * x[:, 1:2 * m:2]
    # Weighted sum — shape (B,)
    s = z @ parity_weights.to(x.device)
    # Threshold at 0. Ties (s==0) have measure zero with continuous weights.
    return (s > 0).long()


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_batch(
    batch_size: int,
    n_cues: int = 7,
    delay: int = 20,
    d_cue: int = 16,
    k: int = 8,
    w: int = 1,
    gap: int = 5,
    noise_level: float = 0.1,
    weights_seed: int = 42,
    seed: Optional[int] = None,
    device: str = "cpu",
    aux_running_readout: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate a batch of enriched evidence accumulation trials.

    Returns
    -------
    inputs  : (T, B, d_cue+3)   float32
    targets : (T, B, 2)         float32  — one-hot at decision step (+ aux if enabled)
    mask    : (T, B)            float32  — 1.0 at decision step, 0 elsewhere
                                          (0.1 at cue-end steps if aux_running_readout)

    where  T = n_cues * (w + gap) + delay + 1
    """
    assert k % 2 == 0, f"k must be even, got {k}"
    assert k <= d_cue, f"k ({k}) must be <= d_cue ({d_cue})"
    assert n_cues >= 1

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    parity_weights = make_parity_weights(k, seed=weights_seed)   # (m,)

    stride = w + gap                     # steps per cue (active + silence)
    cue_window = n_cues * stride
    T = cue_window + delay + 1          # +1 for the single decision step
    t_recall = T - 1
    n_ch = d_cue + 3                    # total input channels
    B = batch_size

    inputs  = torch.zeros(T, B, n_ch)
    targets = torch.zeros(T, B, N_OUT)
    mask    = torch.zeros(T, B)

    # ── Sample binary features for each trial × cue ──────────────────────────
    # binary_features[b, c, f] ∈ {-1, +1}
    raw = torch.randint(0, 2, (B, n_cues, k), generator=gen).float()
    binary_features = raw * 2.0 - 1.0   # (B, n_cues, k)

    # ── Cue labels via parity tree ────────────────────────────────────────────
    cue_labels = torch.stack(
        [parity_label(binary_features[:, c, :], parity_weights) for c in range(n_cues)],
        dim=1,
    )   # (B, n_cues) ∈ {0, 1}

    # ── Build input tensor ────────────────────────────────────────────────────
    for c in range(n_cues):
        for dt in range(w):
            t = c * stride + dt
            # Relevant binary features (±1) + Gaussian noise on all d_cue channels
            pattern = torch.randn(B, d_cue, generator=gen) * noise_level
            pattern[:, :k] = binary_features[:, c, :]
            pattern[:, :k] += torch.randn(B, k, generator=gen) * noise_level
            inputs[t, :, :d_cue] = pattern

    # Recall signal at the single decision step
    inputs[t_recall, :, d_cue] = 1.0

    # Always-on Gaussian noise channel
    inputs[:, :, d_cue + 1] = torch.randn(T, B, generator=gen) * noise_level

    # Constant bias channel
    inputs[:, :, d_cue + 2] = 1.0

    # ── Majority-vote labels ─────────────────────────────────────────────────
    right_count = (cue_labels == 1).sum(dim=1).float()   # (B,)
    left_count  = (cue_labels == 0).sum(dim=1).float()
    labels = (right_count > left_count).long()

    # Ties (rare / absent with odd n_cues + generic parity function)
    tied = right_count == left_count
    if tied.any():
        n_tied = int(tied.sum().item())
        labels[tied] = torch.randint(0, 2, (n_tied,), generator=gen)

    # ── Decision-step targets and mask ────────────────────────────────────────
    targets[t_recall, torch.arange(B), labels] = 1.0
    mask[t_recall] = 1.0

    # ── Optional auxiliary readout at each cue-end step ──────────────────────
    # Enabling this raises early-timestep cosine SNR but dilutes temporal-credit
    # stress.  Keep aux_running_readout=False for headline runs.
    if aux_running_readout:
        for c in range(n_cues):
            t_end = c * stride + w - 1
            right_so_far = (cue_labels[:, : c + 1] == 1).sum(1).float()
            left_so_far  = (cue_labels[:, : c + 1] == 0).sum(1).float()
            diff = (right_so_far - left_so_far) / (c + 1)  # ∈ [-1, +1]
            p_right = torch.sigmoid(diff * 3.0)             # soft target
            targets[t_end, :, 1] = p_right
            targets[t_end, :, 0] = 1.0 - p_right
            mask[t_end] = AUX_MASK_WEIGHT

    return inputs.to(device), targets.to(device), mask.to(device)


# ── Metadata helpers ──────────────────────────────────────────────────────────

def n_in(d_cue: int = 16) -> int:
    """Number of input channels (d_cue + recall + noise + bias)."""
    return d_cue + 3


def sequence_length(n_cues: int = 7, delay: int = 20, w: int = 1, gap: int = 5) -> int:
    """Total sequence length T."""
    return n_cues * (w + gap) + delay + 1


def decision_timestep(n_cues: int = 7, delay: int = 20, w: int = 1, gap: int = 5) -> int:
    """Index of the single decision step (= T - 1)."""
    return sequence_length(n_cues, delay, w, gap) - 1


def chance_accuracy() -> float:
    return 0.5


def task_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correct trials at the decision step.

    Only timesteps where mask > 0.5 (i.e., the full-weight decision step)
    are counted, so the auxiliary cue-end steps are excluded.
    """
    decision = mask > 0.5
    pred    = logits.argmax(dim=-1)   # (T, B)
    tgt     = targets.argmax(dim=-1)  # (T, B)
    correct = ((pred == tgt) * decision).sum().item()
    total   = decision.sum().item()
    return correct / total if total > 0 else 0.0
