"""RNN with optional inserted feedforward layers.

This is the Bellec-style inserted-layer architecture used in the updated plan:
one recurrent tanh layer at the bottom, followed by zero or more non-recurrent
tanh hidden layers, then a linear readout.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class StackedRNN(nn.Module):
    """Single recurrent layer plus configurable feedforward hidden layers."""

    def __init__(
        self,
        n_in: int = 3,
        n_rec: int = 64,
        n_out: int = 2,
        n_ff: int = 1,
        n_ff_hidden: int | None = None,
        spectral_radius: float = 0.9,
    ) -> None:
        super().__init__()
        if n_ff not in (0, 1, 2):
            raise ValueError("n_ff must be one of {0, 1, 2} for the M1 scaffold")
        if n_in <= 0 or n_rec <= 0 or n_out <= 0:
            raise ValueError("n_in, n_rec, and n_out must be positive")

        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_ff = n_ff
        self.n_ff_hidden = n_rec if n_ff_hidden is None else n_ff_hidden
        self.hidden_sizes = [n_rec] + [self.n_ff_hidden for _ in range(n_ff)]

        self.W_in = nn.Parameter(torch.randn(n_rec, n_in) / (n_in ** 0.5))
        self.W_rec = nn.Parameter(self._init_recurrent(n_rec, spectral_radius))
        self.b_rec = nn.Parameter(torch.zeros(n_rec))

        ff_weights = []
        ff_biases = []
        prev_size = n_rec
        for _ in range(n_ff):
            ff_weights.append(
                nn.Parameter(torch.randn(self.n_ff_hidden, prev_size) / (prev_size ** 0.5))
            )
            ff_biases.append(nn.Parameter(torch.zeros(self.n_ff_hidden)))
            prev_size = self.n_ff_hidden
        self.W_ffs = nn.ParameterList(ff_weights)
        self.b_ffs = nn.ParameterList(ff_biases)

        self.W_out = nn.Parameter(torch.randn(n_out, prev_size) / (prev_size ** 0.5))
        self.b_out = nn.Parameter(torch.zeros(n_out))

    @staticmethod
    def _init_recurrent(n_rec: int, spectral_radius: float) -> Tensor:
        W_rec = torch.randn(n_rec, n_rec) / (n_rec ** 0.5)
        with torch.no_grad():
            radius = torch.linalg.eigvals(W_rec).abs().max().real
            if radius > 0:
                W_rec *= spectral_radius / radius
        return W_rec

    def init_hidden(self, batch_size: int, device=None, dtype=None) -> List[Tensor]:
        dev = device or self.W_rec.device
        dt = dtype or self.W_rec.dtype
        return [
            torch.zeros(batch_size, size, device=dev, dtype=dt)
            for size in self.hidden_sizes
        ]

    def step(self, x: Tensor, states: List[Tensor]) -> Tuple[List[Tensor], Tensor]:
        """Run one timestep and return ``(new_states, logits)``."""
        if len(states) != len(self.hidden_sizes):
            raise ValueError(f"expected {len(self.hidden_sizes)} states, got {len(states)}")

        rec_pre = x @ self.W_in.T + states[0] @ self.W_rec.T + self.b_rec
        current = torch.tanh(rec_pre)
        new_states = [current]

        for W_ff, b_ff in zip(self.W_ffs, self.b_ffs):
            current = torch.tanh(current @ W_ff.T + b_ff)
            new_states.append(current)

        logits = current @ self.W_out.T + self.b_out
        return new_states, logits

    def forward(self, inputs: Tensor) -> Tuple[Tensor, List[List[Tensor]]]:
        """Run a sequence.

        Parameters
        ----------
        inputs:
            Tensor of shape ``(T, B, n_in)``.

        Returns
        -------
        logits:
            Tensor of shape ``(T, B, n_out)``.
        all_states:
            Length ``T + 1`` list. Each item is a list containing the recurrent
            state followed by any inserted feedforward-layer activations.
        """
        if inputs.ndim != 3 or inputs.shape[-1] != self.n_in:
            raise ValueError(f"inputs must have shape (T, B, {self.n_in})")

        T, B, _ = inputs.shape
        states = self.init_hidden(B, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        all_states = [states]

        for t in range(T):
            states, logits = self.step(inputs[t], states)
            outputs.append(logits)
            all_states.append(states)

        return torch.stack(outputs, dim=0), all_states
