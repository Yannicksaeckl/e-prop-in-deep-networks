"""
Standalone sanity checks for the revised deep e-prop plan.

Run from the repo root:
    python tests/sanity_checks.py
    python -m tests.sanity_checks

These checks are intentionally small. They verify plumbing and numerical gates;
full training acceptance runs are tracked in PLAN_ALIGNMENT.md.
"""

import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning_rules.bptt import compute_bptt_gradients, _mse_loss, _trace_mse_loss
from learning_rules.deep_eprop import compute_deep_eprop_gradients
from learning_rules.deep_rtrl import compute_deep_rtrl_gradients
from learning_rules.eprop import compute_eprop_gradients, mse_error
from models.deep_rnn import DeepRNN
from models.vanilla_rnn import VanillaRNN
from tasks.cue_accumulation import generate_batch as ca_batch
from tasks.store_and_recall import generate_batch as sr_batch
from tasks.store_and_recall import recall_timestep as sr_recall_timestep
from tasks.store_and_recall import sequence_length as sr_sequence_length
from tasks.store_and_recall import task_accuracy as sr_accuracy
from utils import cosine_sim_grads


def _cosine(g1, g2, keys=None):
    if keys is None:
        keys = [k for k in g1 if k in g2]
    return cosine_sim_grads(g1, g2, keys)


def test_store_and_recall_task() -> bool:
    """Shape, mask, and frozen-model sanity for store-and-recall."""
    print("Test 0a: store_and_recall task sanity ...")

    batch_size, n_patterns, delay = 128, 4, 7
    inputs, targets, mask = sr_batch(
        batch_size=batch_size,
        n_patterns=n_patterns,
        delay=delay,
        cue_duration=2,
        output_duration=3,
    )

    t_expected = sr_sequence_length(delay, cue_duration=2, output_duration=3)
    assert inputs.shape == (t_expected, batch_size, n_patterns + 2), inputs.shape
    assert targets.shape == (t_expected, batch_size, n_patterns), targets.shape
    assert mask.shape == (t_expected, batch_size), mask.shape
    assert mask.sum().item() == float(batch_size * 3), mask.sum().item()
    assert sr_recall_timestep(delay, cue_duration=2) == 2 + delay

    seeded_a = sr_batch(8, n_patterns, delay, seed=123, return_labels=True)
    seeded_b = sr_batch(8, n_patterns, delay, seed=123, return_labels=True)
    for tensor_a, tensor_b in zip(seeded_a, seeded_b):
        assert torch.equal(tensor_a, tensor_b)

    model = VanillaRNN(n_in=n_patterns + 2, n_rec=10, n_out=n_patterns)
    outputs, _ = model(inputs)
    acc = sr_accuracy(outputs, targets, mask)
    assert 0.0 <= acc <= 1.0, acc

    print(f"  PASS  T={t_expected}  mask.sum={int(mask.sum().item())}  untrained_acc={acc:.3f}")
    return True


def test_cue_accumulation_task() -> bool:
    """Shape, mask, label balance, and frozen-model sanity checks."""
    print("Test 0b: cue_accumulation task sanity ...")

    n_cues, delay, batch_size = 5, 20, 512
    inputs, targets, mask = ca_batch(
        batch_size=batch_size,
        n_cues=n_cues,
        delay=delay,
        seed=0,
    )

    t_expected = n_cues * (1 + 5) + delay + 1
    assert inputs.shape == (t_expected, batch_size, 5), inputs.shape
    assert targets.shape == (t_expected, batch_size, 2), targets.shape
    assert mask.shape == (t_expected, batch_size), mask.shape
    assert mask.sum().item() == float(batch_size), mask.sum().item()

    labels = targets[mask.bool()].argmax(dim=-1)
    frac_right = labels.float().mean().item()
    assert 0.40 < frac_right < 0.60, frac_right

    model = VanillaRNN(n_in=5, n_rec=10, n_out=2)
    outputs, _ = model(inputs)
    pred = outputs.argmax(dim=-1)
    target_idx = targets.argmax(dim=-1)
    acc = ((pred == target_idx) * mask).sum().item() / batch_size
    assert 0.10 < acc < 0.90, acc

    print(
        f"  PASS  T={t_expected}  mask.sum={batch_size}  "
        f"frac_right={frac_right:.3f}  untrained_acc={acc:.3f}"
    )
    return True


def test_deep_rtrl_matches_bptt(n_seeds: int = 10) -> bool:
    """G2: deep-RTRL must numerically match BPTT on a tiny 2-layer net."""
    print("Test 1: deep-RTRL == BPTT (absolute/relative error) ...")

    keys = [
        "W_in",
        "W_recs.0",
        "W_recs.1",
        "W_ffs.0",
        "biases.0",
        "biases.1",
        "W_out",
        "b_out",
    ]
    max_abs = 0.0
    max_rel = 0.0

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = DeepRNN(n_in=4, n_rec=8, n_out=2, n_layers=2)
        inputs = torch.randn(8, 4, 4)
        targets = torch.randn(8, 4, 2)
        mask = torch.zeros(8, 4)
        mask[3:6] = 1.0

        g_bptt = compute_bptt_gradients(model, inputs, targets, mask, _trace_mse_loss)
        g_rtrl = compute_deep_rtrl_gradients(model, inputs, targets, mask, mse_error)

        for key in keys:
            if key not in g_bptt or key not in g_rtrl:
                print(f"  FAIL seed={seed}: missing key {key}")
                return False
            diff = (g_rtrl[key] - g_bptt[key]).abs()
            abs_err = diff.max().item()
            rel_err = (diff.norm() / (g_bptt[key].norm() + 1e-12)).item()
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if abs_err > 1e-5 or rel_err > 1e-5:
                print(
                    f"  FAIL seed={seed}, key={key}: "
                    f"max_abs={abs_err:.2e}, rel={rel_err:.2e}"
                )
                return False

    print(f"  PASS  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}")
    return True


def test_depth1_deep_eprop_matches_single(n_seeds: int = 5) -> bool:
    """G1 plumbing: L=1 deep e-prop must equal single-layer e-prop."""
    print("Test 2: depth-1 deep e-prop == single-layer e-prop ...")

    key_map = {
        "W_recs.0": "W_rec",
        "biases.0": "b_rec",
        "W_in": "W_in",
        "W_out": "W_out",
        "b_out": "b_out",
    }

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model_deep = DeepRNN(n_in=4, n_rec=16, n_out=2, n_layers=1)
        model_single = VanillaRNN(n_in=4, n_rec=16, n_out=2)

        with torch.no_grad():
            model_single.W_rec.copy_(model_deep.W_recs[0])
            model_single.W_in.copy_(model_deep.W_in)
            model_single.b_rec.copy_(model_deep.biases[0])
            model_single.W_out.copy_(model_deep.W_out)
            model_single.b_out.copy_(model_deep.b_out)

        inputs = torch.randn(6, 8, 4)
        targets = torch.randn(6, 8, 2)
        mask = torch.zeros(6, 8)
        mask[4] = 1.0

        g_deep = compute_deep_eprop_gradients(
            model_deep, inputs, targets, mask, mse_error, d_zero=False
        )
        g_single = compute_eprop_gradients(
            model_single, inputs, targets, mask, mse_error, d_zero=False
        )

        for deep_key, single_key in key_map.items():
            if not torch.allclose(g_deep[deep_key], g_single[single_key], rtol=1e-5, atol=1e-6):
                diff = (g_deep[deep_key] - g_single[single_key]).abs().max().item()
                print(f"  FAIL seed={seed}, key={deep_key}: max_abs={diff:.2e}")
                return False

    print("  PASS  L=1 deep e-prop matches single-layer e-prop")
    return True


def test_finite_difference_bptt() -> bool:
    """Finite-difference check for BPTT on a small VanillaRNN."""
    print("Test 3: finite-difference gradient check (BPTT) ...")

    torch.manual_seed(1)
    n_rec = 8
    model = VanillaRNN(n_in=3, n_rec=n_rec, n_out=2)
    inputs = torch.randn(5, 3, 3)
    targets = torch.randn(5, 3, 2)
    mask = torch.zeros(5, 3)
    mask[3:5] = 1.0
    eps = 1e-4

    def loss_value():
        with torch.no_grad():
            outputs, _ = model(inputs)
            return _mse_loss(outputs, targets, mask).item()

    g_auto = compute_bptt_gradients(model, inputs, targets, mask)
    g_fd = torch.zeros_like(model.W_rec)

    for i in range(n_rec):
        for j in range(n_rec):
            model.W_rec.data[i, j] += eps
            plus = loss_value()
            model.W_rec.data[i, j] -= 2 * eps
            minus = loss_value()
            model.W_rec.data[i, j] += eps
            g_fd[i, j] = (plus - minus) / (2 * eps)

    rel_err = ((g_auto["W_rec"] - g_fd).norm() / (g_fd.norm() + 1e-12)).item()
    cos = F.cosine_similarity(
        g_auto["W_rec"].flatten().unsqueeze(0),
        g_fd.flatten().unsqueeze(0),
    ).item()

    if rel_err > 3e-2 or cos < 0.99:
        print(f"  FAIL  rel_err={rel_err:.2e}  cosine={cos:.4f}")
        return False

    print(f"  PASS  W_rec rel_err={rel_err:.2e}  cosine={cos:.4f}")
    return True


def test_vanilla_rnn_eprop_approx_d0(n_seeds: int = 5) -> bool:
    """Document that vanilla tanh e-prop and d=0 are close at initialization."""
    print("Test 4: vanilla tanh e-prop approximately d=0 at initialization ...")

    min_cos = 1.0
    for seed in range(n_seeds):
        torch.manual_seed(seed * 7)
        model = VanillaRNN(n_in=4, n_rec=30, n_out=2)
        inputs = torch.randn(12, 16, 4)
        targets = torch.randn(12, 16, 2)
        mask = torch.zeros(12, 16)
        mask[9:12] = 1.0

        g_ep = compute_eprop_gradients(model, inputs, targets, mask, mse_error, d_zero=False)
        g_d0 = compute_eprop_gradients(model, inputs, targets, mask, mse_error, d_zero=True)
        cos = _cosine(g_ep, g_d0)
        if cos == cos:
            min_cos = min(min_cos, cos)

    if min_cos < 0.99:
        print(f"  FAIL  min cosine={min_cos:.4f}")
        return False

    print(f"  PASS  min cosine(e-prop,d=0)={min_cos:.4f}")
    return True


def test_g0_bptt_store_recall_depths() -> bool:
    """G0 smoke: BPTT gradients exist for store-and-recall at depths 1 and 2."""
    print("Test 5: G0 BPTT smoke on store-and-recall depths 1 and 2 ...")

    for depth in [1, 2]:
        torch.manual_seed(depth)
        model = DeepRNN(n_in=6, n_rec=12, n_out=4, n_layers=depth)
        inputs, targets, mask = sr_batch(
            batch_size=8,
            n_patterns=4,
            delay=2,
            cue_duration=1,
            output_duration=1,
        )
        grads = compute_bptt_gradients(model, inputs, targets, mask, _trace_mse_loss)
        required = ["W_in", "W_recs.0", "biases.0", "W_out", "b_out"]
        if depth == 2:
            required.extend(["W_recs.1", "W_ffs.0", "biases.1"])

        missing = [key for key in required if key not in grads]
        if missing:
            print(f"  FAIL depth={depth}: missing gradients {missing}")
            return False
        if any(not torch.isfinite(grads[key]).all() for key in required):
            print(f"  FAIL depth={depth}: non-finite gradient")
            return False

        with torch.no_grad():
            outputs, _ = model(inputs)
        acc = sr_accuracy(outputs, targets, mask)
        if not 0.0 <= acc <= 1.0:
            print(f"  FAIL depth={depth}: invalid accuracy {acc}")
            return False

    print("  PASS  BPTT gradients finite for depth 1 and 2")
    return True


def main():
    print("=" * 60)
    print("Sanity checks - revised deep e-prop plan")
    print("=" * 60)
    print()

    results = {}
    results["0a: store task"] = test_store_and_recall_task()
    print()
    results["0b: cue task"] = test_cue_accumulation_task()
    print()
    results["1: RTRL==BPTT"] = test_deep_rtrl_matches_bptt()
    print()
    results["2: depth-1 eq"] = test_depth1_deep_eprop_matches_single()
    print()
    results["3: FD check"] = test_finite_difference_bptt()
    print()
    results["4: vanilla approx d0"] = test_vanilla_rnn_eprop_approx_d0()
    print()
    results["5: G0 BPTT smoke"] = test_g0_bptt_store_recall_depths()
    print()

    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {name}")
    print()
    if passed == total:
        print(f"All {total} tests passed.")
    else:
        print(f"{passed}/{total} tests passed. See FAIL lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
