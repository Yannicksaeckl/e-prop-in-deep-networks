"""
Reusable store-and-recall experiments for the deep e-prop project.

This module is intentionally Colab-friendly: it exposes small composable
functions that return JSON-serializable dictionaries. Heavy sweeps can run in a
notebook, while local development can call the same functions with tiny configs
as smoke tests.
"""

import json
import os
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learning_rules.bptt import _trace_mse_loss, compute_bptt_gradients
from learning_rules.deep_eprop import compute_deep_eprop_gradients, mse_error
from learning_rules.deep_rtrl import compute_deep_rtrl_gradients
from models.deep_rnn import DeepRNN
from tasks.store_and_recall import (
    chance_accuracy,
    generate_batch,
    input_size,
    output_size,
    task_accuracy,
)

METHOD_LABELS = {
    "bptt": "BPTT",
    "deep-eprop": "Deep e-prop",
    "d=0": "d=0",
}


@dataclass(frozen=True)
class StoreRecallConfig:
    n_patterns: int = 4
    delay: int = 2
    cue_duration: int = 1
    output_duration: int = 1

    @property
    def n_in(self) -> int:
        return input_size(self.n_patterns)

    @property
    def n_out(self) -> int:
        return output_size(self.n_patterns)

    @property
    def chance(self) -> float:
        return chance_accuracy(self.n_patterns)


@dataclass(frozen=True)
class TrainConfig:
    n_rec: int = 64
    n_layers: int = 2
    batch_size: int = 64
    eval_batch_size: int = 256
    n_steps: int = 1000
    eval_every: int = 50
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip_norm: Optional[float] = None


def with_depth(config: TrainConfig, depth: int) -> TrainConfig:
    """Return a copy of ``config`` with a different network depth."""
    return replace(config, n_layers=int(depth))


def with_delay(config: StoreRecallConfig, delay: int) -> StoreRecallConfig:
    """Return a copy of ``config`` with a different task delay."""
    return replace(config, delay=int(delay))


def normalize_method(method: str) -> str:
    """Canonicalize method aliases used in notebooks/scripts."""
    key = method.lower().replace("_", "-").strip()
    aliases = {
        "deep-e-prop": "deep-eprop",
        "deepeprop": "deep-eprop",
        "eprop": "deep-eprop",
        "deep eprop": "deep-eprop",
        "d0": "d=0",
        "d-zero": "d=0",
        "zero": "d=0",
    }
    key = aliases.get(key, key)
    if key not in METHOD_LABELS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHOD_LABELS)}")
    return key


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable sweeps."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def new_model(task: StoreRecallConfig, train: TrainConfig, seed: int) -> DeepRNN:
    """Create a freshly seeded DeepRNN for store-and-recall."""
    seed_everything(seed)
    model = DeepRNN(task.n_in, train.n_rec, task.n_out, n_layers=train.n_layers)
    return model.to(train.device)


def batch_kwargs(task: StoreRecallConfig) -> Dict[str, int]:
    """Common keyword args for ``tasks.store_and_recall.generate_batch``."""
    return {
        "n_patterns": task.n_patterns,
        "delay": task.delay,
        "cue_duration": task.cue_duration,
        "output_duration": task.output_duration,
    }


def gradients_for_method(
    model: DeepRNN,
    method: str,
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Dict[str, Tensor]:
    """Compute BPTT, deep e-prop, or d=0 gradients for ``model``."""
    method = normalize_method(method)
    if method == "bptt":
        return compute_bptt_gradients(model, inputs, targets, mask, _trace_mse_loss)
    return compute_deep_eprop_gradients(
        model,
        inputs,
        targets,
        mask,
        mse_error,
        d_zero=(method == "d=0"),
    )


def gradient_norm(grads: Dict[str, Tensor]) -> float:
    """Return the global L2 norm of a gradient dictionary."""
    if not grads:
        return 0.0
    total = torch.zeros((), device=next(iter(grads.values())).device)
    for grad in grads.values():
        total = total + (grad.detach() ** 2).sum()
    return float(torch.sqrt(total).item())


def apply_grads(
    model: DeepRNN,
    grads: Dict[str, Tensor],
    lr: float,
    clip_norm: Optional[float] = None,
) -> float:
    """Apply a gradient dictionary with SGD and optional global norm clipping."""
    norm = gradient_norm(grads)
    scale = 1.0
    if clip_norm is not None and norm > clip_norm:
        scale = float(clip_norm) / (norm + 1e-12)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in grads:
                param.add_(grads[name], alpha=-float(lr) * scale)
    return norm


def evaluate_model(
    model: DeepRNN,
    task: StoreRecallConfig,
    batch_size: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate loss and accuracy on a deterministic held-out batch."""
    inputs, targets, mask = generate_batch(
        batch_size=batch_size,
        device=device,
        seed=seed,
        **batch_kwargs(task),
    )
    with torch.no_grad():
        outputs, _ = model(inputs)
        loss = _trace_mse_loss(outputs, targets, mask)
        acc = task_accuracy(outputs, targets, mask)
    return {"loss": float(loss.item()), "accuracy": float(acc)}


def layer_parameter_keys(n_layers: int) -> Dict[str, List[str]]:
    """Return parameter groups for layer-wise gradient cosine diagnostics."""
    groups: Dict[str, List[str]] = {}
    hidden_all: List[str] = []
    for layer in range(n_layers):
        keys = [f"W_recs.{layer}", f"biases.{layer}"]
        if layer == 0:
            keys.append("W_in")
        else:
            keys.append(f"W_ffs.{layer - 1}")
        groups[f"layer{layer + 1}"] = keys
        hidden_all.extend(keys)

    groups["hidden_all"] = hidden_all
    groups["readout"] = ["W_out", "b_out"]
    groups["all"] = hidden_all + groups["readout"]
    if n_layers >= 1:
        groups["bottom"] = groups["layer1"]
        groups["top"] = groups[f"layer{n_layers}"]
    return groups


def cosine_for_keys(
    g_approx: Dict[str, Tensor],
    g_ref: Dict[str, Tensor],
    keys: Sequence[str],
    eps: float = 1e-12,
) -> float:
    """Cosine similarity for selected keys shared by two gradient dicts."""
    parts_a: List[Tensor] = []
    parts_b: List[Tensor] = []
    for key in keys:
        if key in g_approx and key in g_ref:
            parts_a.append(g_approx[key].detach().flatten())
            parts_b.append(g_ref[key].detach().flatten())
    if not parts_a:
        return float("nan")

    vec_a = torch.cat(parts_a)
    vec_b = torch.cat(parts_b)
    norm_a = vec_a.norm().item()
    norm_b = vec_b.norm().item()
    if norm_a < eps or norm_b < eps:
        return float("nan")
    return float((vec_a @ vec_b / (norm_a * norm_b)).item())


def gradient_alignment(
    model: DeepRNN,
    task: StoreRecallConfig,
    batch_size: int,
    seed: int,
    methods: Sequence[str] = ("deep-eprop", "d=0"),
) -> Dict[str, Dict[str, float]]:
    """Compare approximate gradients to BPTT by layer on one batch."""
    inputs, targets, mask = generate_batch(
        batch_size=batch_size,
        device=next(model.parameters()).device,
        seed=seed,
        **batch_kwargs(task),
    )
    g_bptt = gradients_for_method(model, "bptt", inputs, targets, mask)
    groups = layer_parameter_keys(model.n_layers)

    result: Dict[str, Dict[str, float]] = {}
    for method in methods:
        method = normalize_method(method)
        if method == "bptt":
            continue
        g_approx = gradients_for_method(model, method, inputs, targets, mask)
        result[method] = {
            group: cosine_for_keys(g_approx, g_bptt, keys)
            for group, keys in groups.items()
        }
    return result


def train_once(
    method: str,
    seed: int,
    task: Optional[StoreRecallConfig] = None,
    train: Optional[TrainConfig] = None,
    track_alignment: bool = False,
    alignment_methods: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """Train one model and return a JSON-serializable learning curve."""
    method = normalize_method(method)
    task = task or StoreRecallConfig()
    train = train or TrainConfig()
    alignment_methods = alignment_methods or (method,)

    model = new_model(task, train, seed)
    steps: List[int] = []
    eval_accuracy: List[float] = []
    eval_loss: List[float] = []
    grad_norms: List[float] = []
    alignment_records: List[Dict[str, object]] = []

    def record(step: int) -> None:
        metrics = evaluate_model(
            model,
            task,
            batch_size=train.eval_batch_size,
            seed=10_000_000 + seed * 100_003 + step,
            device=train.device,
        )
        steps.append(int(step))
        eval_accuracy.append(metrics["accuracy"])
        eval_loss.append(metrics["loss"])

        if track_alignment:
            align = gradient_alignment(
                model,
                task,
                batch_size=min(train.eval_batch_size, train.batch_size),
                seed=20_000_000 + seed * 100_003 + step,
                methods=alignment_methods,
            )
            for approx_method, by_layer in align.items():
                for layer, cosine in by_layer.items():
                    alignment_records.append(
                        {
                            "step": int(step),
                            "trained_method": method,
                            "approx_method": approx_method,
                            "layer": layer,
                            "cosine": float(cosine),
                        }
                    )

    record(0)
    if verbose:
        print(f"[{METHOD_LABELS[method]}] seed={seed} step=0 acc={eval_accuracy[-1]:.3f}")

    for step in range(1, train.n_steps + 1):
        inputs, targets, mask = generate_batch(
            batch_size=train.batch_size,
            device=train.device,
            seed=seed * 1_000_003 + step,
            **batch_kwargs(task),
        )
        grads = gradients_for_method(model, method, inputs, targets, mask)
        grad_norms.append(apply_grads(model, grads, train.lr, train.grad_clip_norm))

        if step % train.eval_every == 0 or step == train.n_steps:
            record(step)
            if verbose:
                print(
                    f"[{METHOD_LABELS[method]}] seed={seed} "
                    f"step={step} acc={eval_accuracy[-1]:.3f} loss={eval_loss[-1]:.4f}"
                )

    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": int(seed),
        "task": asdict(task),
        "train": asdict(train),
        "steps": steps,
        "eval_accuracy": eval_accuracy,
        "eval_loss": eval_loss,
        "final_accuracy": eval_accuracy[-1],
        "final_loss": eval_loss[-1],
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "alignment": alignment_records,
    }


def _stderr(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1:], dtype=float)
    return values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _group_runs(runs: Sequence[Dict[str, object]]) -> Dict[Tuple[str, int, int], List[Dict[str, object]]]:
    groups: Dict[Tuple[str, int, int], List[Dict[str, object]]] = {}
    for run in runs:
        method = str(run["method"])
        depth = int(run["train"]["n_layers"])
        delay = int(run["task"]["delay"])
        groups.setdefault((method, depth, delay), []).append(run)
    return groups


def summarize_learning_runs(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate learning curves by method, depth, and task delay."""
    summaries: List[Dict[str, object]] = []
    for (method, depth, delay), group in _group_runs(runs).items():
        steps = list(group[0]["steps"])
        acc = np.asarray([r["eval_accuracy"] for r in group], dtype=float)
        loss = np.asarray([r["eval_loss"] for r in group], dtype=float)
        final_acc = acc[:, -1]
        final_loss = loss[:, -1]
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "depth": int(depth),
                "delay": int(delay),
                "n_seeds": int(len(group)),
                "steps": steps,
                "accuracy_mean": acc.mean(axis=0).tolist(),
                "accuracy_stderr": _stderr(acc).tolist(),
                "loss_mean": loss.mean(axis=0).tolist(),
                "loss_stderr": _stderr(loss).tolist(),
                "final_accuracy_mean": float(final_acc.mean()),
                "final_accuracy_stderr": float(final_acc.std(ddof=1) / np.sqrt(len(final_acc)))
                if len(final_acc) > 1
                else 0.0,
                "final_loss_mean": float(final_loss.mean()),
                "final_loss_stderr": float(final_loss.std(ddof=1) / np.sqrt(len(final_loss)))
                if len(final_loss) > 1
                else 0.0,
            }
        )
    return sorted(summaries, key=lambda x: (x["depth"], x["delay"], x["method"]))


def summarize_alignment_records(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate alignment records emitted by ``train_once``."""
    records: List[Dict[str, object]] = []
    for run in runs:
        for record in run.get("alignment", []):
            merged = dict(record)
            merged["seed"] = run["seed"]
            merged["depth"] = run["train"]["n_layers"]
            merged["delay"] = run["task"]["delay"]
            records.append(merged)

    groups: Dict[Tuple[object, ...], List[float]] = {}
    for record in records:
        key = (
            record["trained_method"],
            record["approx_method"],
            record["depth"],
            record["delay"],
            record["step"],
            record["layer"],
        )
        groups.setdefault(key, []).append(float(record["cosine"]))

    summaries: List[Dict[str, object]] = []
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        trained_method, approx_method, depth, delay, step, layer = key
        summaries.append(
            {
                "trained_method": trained_method,
                "approx_method": approx_method,
                "depth": int(depth),
                "delay": int(delay),
                "step": int(step),
                "layer": layer,
                "cosine_mean": float(arr.mean()),
                "cosine_stderr": stderr,
                "n_seeds": int(len(arr)),
            }
        )
    return sorted(
        summaries,
        key=lambda x: (x["trained_method"], x["approx_method"], x["layer"], x["step"]),
    )


def run_learning_curves(
    methods: Sequence[str] = ("bptt", "deep-eprop", "d=0"),
    seeds: Sequence[int] = (0,),
    task: Optional[StoreRecallConfig] = None,
    train: Optional[TrainConfig] = None,
    track_alignment_for: Sequence[str] = (),
    verbose: bool = False,
) -> Dict[str, object]:
    """Run multi-seed learning curves for a fixed task/depth config."""
    task = task or StoreRecallConfig()
    train = train or TrainConfig()
    track_set = {normalize_method(method) for method in track_alignment_for}

    runs: List[Dict[str, object]] = []
    for method in methods:
        method = normalize_method(method)
        for seed in seeds:
            runs.append(
                train_once(
                    method,
                    seed=int(seed),
                    task=task,
                    train=train,
                    track_alignment=(method in track_set),
                    alignment_methods=(method,),
                    verbose=verbose,
                )
            )

    return {
        "runs": runs,
        "summary": summarize_learning_runs(runs),
        "alignment_summary": summarize_alignment_records(runs),
    }


def run_delay_training_sweep(
    delays: Sequence[int],
    methods: Sequence[str] = ("bptt", "deep-eprop", "d=0"),
    seeds: Sequence[int] = (0,),
    task: Optional[StoreRecallConfig] = None,
    train: Optional[TrainConfig] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """Train every method at each delay and summarize final performance."""
    base_task = task or StoreRecallConfig()
    train = train or TrainConfig()
    runs: List[Dict[str, object]] = []
    for delay in delays:
        task_for_delay = with_delay(base_task, int(delay))
        result = run_learning_curves(
            methods=methods,
            seeds=seeds,
            task=task_for_delay,
            train=train,
            verbose=verbose,
        )
        runs.extend(result["runs"])
    return {
        "runs": runs,
        "summary": summarize_learning_runs(runs),
    }


def run_gradient_delay_sweep(
    delays: Sequence[int],
    methods: Sequence[str] = ("deep-eprop", "d=0"),
    n_trials: int = 20,
    seed: int = 0,
    task: Optional[StoreRecallConfig] = None,
    train: Optional[TrainConfig] = None,
) -> Dict[str, object]:
    """Measure untrained gradient cosine vs BPTT across delay values."""
    base_task = task or StoreRecallConfig()
    train = train or TrainConfig()
    records: List[Dict[str, object]] = []

    for delay in delays:
        task_for_delay = with_delay(base_task, int(delay))
        for trial in range(n_trials):
            trial_seed = int(seed) + int(delay) * 10_000 + trial
            model = new_model(task_for_delay, train, trial_seed)
            align = gradient_alignment(
                model,
                task_for_delay,
                batch_size=train.batch_size,
                seed=30_000_000 + trial_seed,
                methods=methods,
            )
            for method, by_layer in align.items():
                for layer, cosine in by_layer.items():
                    records.append(
                        {
                            "delay": int(delay),
                            "trial": int(trial),
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "layer": layer,
                            "cosine": float(cosine),
                        }
                    )

    summary = summarize_gradient_records(records)
    return {"records": records, "summary": summary}


def summarize_gradient_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate records from ``run_gradient_delay_sweep``."""
    groups: Dict[Tuple[object, ...], List[float]] = {}
    for record in records:
        key = (record["method"], record["delay"], record["layer"])
        groups.setdefault(key, []).append(float(record["cosine"]))

    summaries: List[Dict[str, object]] = []
    for (method, delay, layer), vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "delay": int(delay),
                "layer": layer,
                "cosine_mean": float(arr.mean()),
                "cosine_stderr": stderr,
                "n_trials": int(len(arr)),
            }
        )
    return sorted(summaries, key=lambda x: (x["delay"], x["method"], x["layer"]))


def run_rtrl_gate(
    n_reps: int = 5,
    seed: int = 0,
    task: Optional[StoreRecallConfig] = None,
    n_rec: int = 8,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, object]:
    """Small deep-RTRL == BPTT numerical gate for the 2-layer DeepRNN."""
    task = task or StoreRecallConfig(delay=2)
    train = TrainConfig(
        n_rec=n_rec,
        n_layers=2,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        n_steps=1,
        eval_every=1,
        device=device,
    )
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

    records: List[Dict[str, object]] = []
    max_abs = 0.0
    max_rel = 0.0
    for rep in range(n_reps):
        rep_seed = int(seed) + rep
        model = new_model(task, train, rep_seed)
        inputs, targets, mask = generate_batch(
            batch_size=batch_size,
            device=device,
            seed=40_000_000 + rep_seed,
            **batch_kwargs(task),
        )
        g_bptt = gradients_for_method(model, "bptt", inputs, targets, mask)
        g_rtrl = compute_deep_rtrl_gradients(model, inputs, targets, mask, mse_error)

        for key in keys:
            diff = (g_rtrl[key] - g_bptt[key]).detach()
            abs_err = float(diff.abs().max().item())
            rel_err = float(diff.norm().item() / (g_bptt[key].detach().norm().item() + 1e-12))
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            records.append(
                {
                    "rep": int(rep),
                    "key": key,
                    "max_abs": abs_err,
                    "rel_err": rel_err,
                }
            )

    return {
        "records": records,
        "max_abs": float(max_abs),
        "max_rel": float(max_rel),
        "passed": bool(max_abs <= 1e-5 and max_rel <= 1e-5),
    }


def save_json(payload: Dict[str, object], path: os.PathLike) -> None:
    """Write a result dictionary as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: os.PathLike) -> Dict[str, object]:
    """Read a result dictionary written by ``save_json``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    smoke_task = StoreRecallConfig(delay=1)
    smoke_train = TrainConfig(
        n_rec=8,
        n_layers=2,
        batch_size=4,
        eval_batch_size=8,
        n_steps=2,
        eval_every=1,
        device="cpu",
    )
    smoke = run_learning_curves(
        methods=("bptt", "deep-eprop", "d=0"),
        seeds=(0,),
        task=smoke_task,
        train=smoke_train,
        track_alignment_for=("deep-eprop", "d=0"),
        verbose=True,
    )
    gate = run_rtrl_gate(n_reps=1, task=smoke_task, n_rec=4, batch_size=2)
    print(json.dumps({"learning_summary": smoke["summary"], "rtrl_gate": gate}, indent=2))
