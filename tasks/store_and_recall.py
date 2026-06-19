"""
Store-and-recall (delayed copy) task.

Each trial presents a one-hot cue pattern, waits through a blank delay, then
raises a recall signal. The network must reproduce the stored cue during the
recall/output window. The loss mask is zero everywhere except that window, so
the delay is the isolated memory/credit-assignment knob.

Input channels
--------------
0 .. n_patterns-1 : one-hot cue, active during the cue window
n_patterns        : recall signal, active at the first output timestep
n_patterns + 1    : bias, always 1

Target
------
One-hot index of the stored pattern, active during the output window.
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

N_EXTRA_INPUTS = 2


def input_size(n_patterns: int) -> int:
    """Return input dimensionality for ``n_patterns`` cue classes."""
    if n_patterns < 2:
        raise ValueError("n_patterns must be at least 2")
    return n_patterns + N_EXTRA_INPUTS


def output_size(n_patterns: int) -> int:
    """Return output dimensionality for ``n_patterns`` cue classes."""
    if n_patterns < 2:
        raise ValueError("n_patterns must be at least 2")
    return n_patterns


def recall_timestep(delay: int, cue_duration: int = 1) -> int:
    """Return the first timestep where the recall signal and target appear."""
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if cue_duration < 1:
        raise ValueError("cue_duration must be at least 1")
    return cue_duration + delay


def sequence_length(
    delay: int,
    cue_duration: int = 1,
    output_duration: int = 1,
) -> int:
    """Return total sequence length T for the given timing parameters."""
    if output_duration < 1:
        raise ValueError("output_duration must be at least 1")
    return recall_timestep(delay, cue_duration) + output_duration


def chance_accuracy(n_patterns: int) -> float:
    """Chance-level accuracy for uniformly sampled cue classes."""
    return 1.0 / float(output_size(n_patterns))


def generate_batch(
    batch_size: int,
    n_patterns: int,
    delay: int,
    cue_duration: int = 1,
    output_duration: int = 1,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    return_labels: bool = False,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Return a batch of store-and-recall trials.

    Parameters
    ----------
    batch_size      : number of independent trials
    n_patterns      : number of one-hot cue classes
    delay           : blank timesteps between cue window and recall
    cue_duration    : number of timesteps the cue is shown
    output_duration : number of masked output timesteps
    device          : target device for returned tensors
    seed            : optional deterministic label seed
    return_labels   : if True, append the integer cue labels to the return tuple

    Returns
    -------
    inputs  : (T, B, n_patterns + 2)
    targets : (T, B, n_patterns)
    mask    : (T, B), 1.0 where loss is computed
    labels  : (B,), only when return_labels=True
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    device_obj = torch.device(device)
    n_in = input_size(n_patterns)
    n_out = output_size(n_patterns)
    T = sequence_length(delay, cue_duration, output_duration)

    inputs = torch.zeros(T, batch_size, n_in, device=device_obj)
    targets = torch.zeros(T, batch_size, n_out, device=device_obj)
    mask = torch.zeros(T, batch_size, device=device_obj)

    if seed is None:
        labels = torch.randint(0, n_patterns, (batch_size,), device=device_obj)
    else:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        labels = torch.randint(0, n_patterns, (batch_size,), generator=gen).to(device_obj)

    batch_idx = torch.arange(batch_size, device=device_obj)

    for t in range(cue_duration):
        inputs[t, batch_idx, labels] = 1.0

    recall_t = recall_timestep(delay, cue_duration)
    inputs[recall_t, :, n_patterns] = 1.0
    inputs[:, :, n_patterns + 1] = 1.0

    for t in range(recall_t, recall_t + output_duration):
        targets[t, batch_idx, labels] = 1.0
        mask[t] = 1.0

    if return_labels:
        return inputs, targets, mask, labels
    return inputs, targets, mask


def task_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correct argmax predictions over masked timesteps."""
    pred = logits.argmax(dim=-1)
    tgt = targets.argmax(dim=-1)
    correct = ((pred == tgt) * mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0
