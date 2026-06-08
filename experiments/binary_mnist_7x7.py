"""
Binary 7x7 sequential MNIST sanity experiment.

This is an easier sMNIST variant for checking whether BPTT can learn before
using the task as an e-prop diagnostic.

Run:
    python -m experiments.binary_mnist_7x7
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from learning_rules.bptt import _xent_loss
from learning_rules.deep_eprop import compute_deep_eprop_gradients, xent_error
from models.deep_rnn import DeepRNN
from tasks.smnist import generate_batch, task_accuracy


SEED = 42
IMAGE_SIZE = 7
DIGITS = (0, 1)
N_IN = 1
N_OUT = len(DIGITS)
N_REC = 64
N_LAYERS = 1
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
N_STEPS = 1000
EVAL_EVERY = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def batch(batch_size: int, train: bool = True):
    return generate_batch(
        batch_size=batch_size,
        device=DEVICE,
        train=train,
        image_size=IMAGE_SIZE,
        digits=DIGITS,
    )


def bptt_grads(model, inputs, targets, mask):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    outputs, _ = model(inputs)
    _xent_loss(outputs, targets, mask).backward()
    return {
        k: p.grad.clone()
        for k, p in model.named_parameters()
        if p.grad is not None
    }


def apply_grads_adam(model, grads, optimizer):
    optimizer.zero_grad(set_to_none=True)
    for name, param in model.named_parameters():
        if name in grads:
            param.grad = grads[name].detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


def evaluate(model):
    with torch.no_grad():
        inputs, targets, mask = batch(EVAL_BATCH_SIZE, train=False)
        outputs, _ = model(inputs)
        loss = _xent_loss(outputs, targets, mask).item()
        acc = task_accuracy(outputs, targets, mask)
    return loss, acc


def cosine_all_params(g_approx, g_bptt):
    sims = []
    for key in g_bptt:
        if key not in g_approx:
            continue
        v1 = g_approx[key].flatten()
        v2 = g_bptt[key].flatten()
        if v1.norm() < 1e-12 or v2.norm() < 1e-12:
            continue
        sims.append(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
    return float(torch.tensor(sims).mean()) if sims else float("nan")


def gradient_cosines(n_trials: int = 10):
    eprop_sims = []
    d0_sims = []
    for trial in range(n_trials):
        torch.manual_seed(SEED + 1000 + trial)
        model = DeepRNN(N_IN, N_REC, N_OUT, n_layers=N_LAYERS).to(DEVICE)
        inputs, targets, mask = batch(BATCH_SIZE, train=True)
        g_bptt = bptt_grads(model, inputs, targets, mask)
        g_eprop = compute_deep_eprop_gradients(
            model, inputs, targets, mask, xent_error, d_zero=False)
        g_d0 = compute_deep_eprop_gradients(
            model, inputs, targets, mask, xent_error, d_zero=True)
        eprop_sims.append(cosine_all_params(g_eprop, g_bptt))
        d0_sims.append(cosine_all_params(g_d0, g_bptt))
    return sum(eprop_sims) / len(eprop_sims), sum(d0_sims) / len(d0_sims)


def train(label: str, use_bptt: bool, d_zero: bool = False):
    torch.manual_seed(SEED)
    model = DeepRNN(N_IN, N_REC, N_OUT, n_layers=N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    accs = []
    losses = []

    for step in range(N_STEPS + 1):
        inputs, targets, mask = batch(BATCH_SIZE, train=True)
        if use_bptt:
            grads = bptt_grads(model, inputs, targets, mask)
        else:
            grads = compute_deep_eprop_gradients(
                model, inputs, targets, mask, xent_error, d_zero=d_zero)
        apply_grads_adam(model, grads, optimizer)

        if step % EVAL_EVERY == 0:
            loss, acc = evaluate(model)
            losses.append(loss)
            accs.append(acc)
            print(f"  [{label}] step {step:4d}  test_loss={loss:.3f}  test_acc={acc:.3f}")

    return losses, accs


def main():
    os.makedirs("results", exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Task: binary {IMAGE_SIZE}x{IMAGE_SIZE} sMNIST, digits={DIGITS}, T={IMAGE_SIZE * IMAGE_SIZE}")

    print("\n=== Gradient cosine on untrained models ===")
    cos_e, cos_d0 = gradient_cosines()
    print(f"  e-prop vs BPTT: {cos_e:.3f}")
    print(f"  d=0    vs BPTT: {cos_d0:.3f}")

    print("\n=== Learning curves ===")
    curves = {
        "BPTT": train("BPTT", use_bptt=True),
        "e-prop": train("e-prop", use_bptt=False, d_zero=False),
        "d=0": train("d=0", use_bptt=False, d_zero=True),
    }

    steps = list(range(0, N_STEPS + 1, EVAL_EVERY))
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (_, accs) in curves.items():
        ax.plot(steps, accs, marker="o", markersize=3, label=label)
    ax.axhline(0.5, color="gray", linestyle="--", label="chance")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Binary 7x7 sMNIST")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/binary_mnist_7x7_learning_curves.pdf")
    fig.savefig("results/binary_mnist_7x7_learning_curves.svg")
    print("Saved results/binary_mnist_7x7_learning_curves.pdf / .svg")


if __name__ == "__main__":
    main()
