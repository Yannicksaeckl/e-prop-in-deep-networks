"""
Sequential MNIST variants.

The default is standard sMNIST: each 28x28 MNIST image is flattened to
T=784 scalar timesteps, with a classification loss only at the final step.

The same generator can also create easier diagnostic tasks, e.g. binary
7x7 MNIST with T=49:

    generate_batch(batch_size, image_size=7, digits=(0, 1))

psMNIST applies a fixed pixel permutation, removing local spatial structure.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence, Tuple

T_SMNIST = 784
N_CLASSES = 10
DEFAULT_DATA_ROOT = "/tmp/mnist_data"

_cache: dict = {}
_perm_cache: dict = {}


def _permutation(length: int) -> Tensor:
    if length not in _perm_cache:
        generator = torch.Generator().manual_seed(12345)
        _perm_cache[length] = torch.randperm(length, generator=generator)
    return _perm_cache[length]


def _load_mnist(
    train: bool = True,
    image_size: int = 28,
    digits: Optional[Sequence[int]] = None,
    data_root: str = DEFAULT_DATA_ROOT,
):
    digit_tuple = tuple(digits) if digits is not None else None
    key = (train, image_size, digit_tuple, data_root)
    if key not in _cache:
        try:
            import torchvision
        except ImportError as exc:
            raise ImportError("torchvision required: pip install torchvision") from exc

        ds = torchvision.datasets.MNIST(
            root=data_root,
            train=train,
            download=True,
        )
        imgs = ds.data.float() / 255.0
        lbls = ds.targets.long()

        if digit_tuple is None:
            n_classes = N_CLASSES
        else:
            if len(digit_tuple) < 2:
                raise ValueError("digits must contain at least two classes")
            keep = torch.zeros_like(lbls, dtype=torch.bool)
            mapped = torch.empty_like(lbls)
            for class_idx, digit in enumerate(digit_tuple):
                digit_mask = lbls == digit
                keep |= digit_mask
                mapped[digit_mask] = class_idx
            imgs = imgs[keep]
            lbls = mapped[keep]
            n_classes = len(digit_tuple)

        if image_size != 28:
            imgs = F.interpolate(
                imgs.unsqueeze(1),
                size=(image_size, image_size),
                mode="area",
            ).squeeze(1)

        seqs = imgs.reshape(-1, image_size * image_size)
        _cache[key] = (seqs, lbls, n_classes)
    return _cache[key]


def generate_batch(
    batch_size: int,
    permuted: bool = False,
    device: str = "cpu",
    train: bool = True,
    image_size: int = 28,
    digits: Optional[Sequence[int]] = None,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample a random mini-batch for sMNIST-like tasks.

    Returns
    -------
    inputs  : (T, B, 1)     pixel values in [0, 1]
    targets : (T, B, C)     one-hot class label at final timestep, zeros elsewhere
    mask    : (T, B)        1 at final timestep only
    """
    imgs, lbls, n_classes = _load_mnist(
        train=train,
        image_size=image_size,
        digits=digits,
        data_root=data_root,
    )
    idx = torch.randint(0, imgs.shape[0], (batch_size,))
    seqs = imgs[idx]

    if permuted:
        seqs = seqs[:, _permutation(seqs.shape[1])]

    inputs = seqs.T.unsqueeze(-1).to(device)
    T = inputs.shape[0]

    targets = torch.zeros(T, batch_size, n_classes, device=device)
    targets[-1] = F.one_hot(lbls[idx], n_classes).float().to(device)

    mask = torch.zeros(T, batch_size, device=device)
    mask[-1] = 1.0

    return inputs, targets, mask


def task_accuracy(outputs: Tensor, targets: Tensor, mask: Tensor) -> float:
    """Fraction of correctly classified digits at the final timestep."""
    pred = outputs[-1].argmax(-1)
    label = targets[-1].argmax(-1)
    return (pred == label).float().mean().item()
