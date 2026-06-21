# Plan Alignment Status

This file tracks the codebase against `project_plan.md`. Tasks are intentionally
sequential: later experiments should not be treated as done just because old
prototype files or result figures exist.

## Done Already

- `models/deep_rnn.py` implements the required stacked vanilla tanh RNN:
  every layer is recurrent, and upper layers receive same-timestep input from
  the lower layer.
- `tasks/store_and_recall.py` implements the encode-delay-recall task with loss
  masked to the recall window.
- `tasks/cue_accumulation.py` implements the cue-delay-decision task and exposes
  delay as an isolated knob.
- `learning_rules/bptt.py` computes autograd/BPTT gradients for named model
  parameters.
- `learning_rules/bptt.py::_trace_mse_loss` now provides the BPTT loss
  normalization that exactly matches the online trace-rule convention.
- `learning_rules/eprop.py` implements single-layer tanh e-prop and d=0.
- `learning_rules/deep_eprop.py` implements deep e-prop and d=0 for `DeepRNN`.
- `learning_rules/deep_rtrl.py` implements the 2-layer deep-RTRL gate.
- `tests/sanity_checks.py` now checks:
  - store-and-recall and cue-accumulation task shapes/masks,
  - deep-RTRL equals BPTT by absolute/relative error,
  - depth-1 deep e-prop equals the single-layer implementation,
  - BPTT finite-difference sanity,
  - finite BPTT gradients at depths 1 and 2.

## Not Done Yet

- G0 acceptance is not complete: BPTT still needs a clean training run showing
  store-and-recall is solved at depths 1 and 2.
- G1 acceptance is not complete: single-layer e-prop needs a current rerun that
  demonstrates it tracks BPTT on store-and-recall.
- G3 is not complete: depth-2 deep e-prop still needs the plan-aligned core
  comparison against BPTT on store-and-recall, including bottom-layer cosine.
- The delay sweep is not complete: `{BPTT, deep e-prop, d=0}` must be run across
  increasing delay with delay as the only swept task variable.
- Cue accumulation has not yet been repeated with the full core comparison and
  delay sweep.
- Store-and-recall seed robustness, error-bar aggregation, and training-step
  gradient cosine diagnostics are now implemented in
  `experiments/store_recall_suite.py`; the heavy Colab run still needs to be
  executed before treating G0/G1/G3 as accepted.
- Time-timestep-resolved gradient cosine diagnostics are not yet implemented;
  the current store-and-recall diagnostics are resolved by training step and
  layer.
- Depth 3 is optional and should wait until G0-G4 at depth 2 are in good shape.

## G5 — Enriched Evidence Accumulation (depth-loading experiment)

Milestone added for the enriched cue accumulation experiment. Acceptance criteria:

| Gate | Criterion | Status |
|---|---|---|
| G5-depth | BPTT depth=2 decision accuracy > depth=1 by ≥ threshold, stable across ≥3 seeds | Pending (auto-escalating k) |
| G5-rtrl | deep-RTRL == BPTT to ≤ 1e-5 on enriched task | Pending |
| G5-train | All four rules trained at fixed delay; no_eps_z fails while deep-eprop learns | Pending |
| G5-cosine | Layer-resolved cosine vs D: deep-eprop bottom ≠ no_eps_z (separation visible) | Pending |
| G5-decomp | RTRL-minus-eps_z frac_epsz and cos_epsz vs D reported at bottom layer | Pending |

New files:
- `tasks/enriched_evidence_accumulation.py` — parity-tree cue task (degree-2 boolean labels)
- `notebooks/enriched_eprop_colab.ipynb` — full experiment notebook

Note on `no_eps_z`: zeroes `eps_cross_*` in `deep_eprop.py` (all lower-layer gradients come
from this term). Distinct from `d=0` (which drops temporal carry but keeps spatial eps_cross).

## Sequential Work Order

1. Finish G0 on store-and-recall: BPTT training at depth 1 and 2.
2. Finish G1: single-layer e-prop vs BPTT on store-and-recall.
3. Run G2 as a gate before trusting any deep trace result.
4. Finish G3: depth-2 deep e-prop vs BPTT on store-and-recall, with
   bottom-layer cosine emphasized.
5. Add the store-and-recall delay sweep for `{BPTT, deep e-prop, d=0}`.
6. Repeat the core comparison and delay sweep on cue accumulation.
7. Add seed robustness/error bars for the headline comparison.
8. Consider depth 3 only after the above path is stable.

## Redundant Or Out-Of-Scope Files

These files are not necessarily wrong; they are just outside the revised core
plan and should not be used as evidence for the sequential milestones above.

| File or folder | Reason |
|---|---|
| `models/lif_rnn.py` | LIF/ALIF models are explicitly out of scope. |
| `models/deep_lif.py` | Spiking deep LIF is out of scope. |
| `models/deep_alif.py` | ALIF is explicitly out of scope. |
| `models/leaky_rnn.py` | Leaky alpha sweeps are outside the stacked vanilla tanh plan. |
| `models/vanilla_rnn.py::LeakyRNN` | Duplicate/legacy leaky implementation; redundant with `models/leaky_rnn.py` and out of scope. |
| `learning_rules/eprop.py::compute_eprop_leaky_gradients` | Supports the old leaky branch, not the revised core result. |
| `run_exp5.sh` | Points to an old experiment numbering path; not part of the revised sequence. |
| `_slim_notebook.py` | Notebook surgery/prototype helper; not part of the core experiment path. |
| `deep_eprop_colab.ipynb` | Contains old leaky/LIF/ALIF/sMNIST/SHD material; needs pruning before use as the current project notebook. |
| `results/exp4_*` | Leaky-RNN alpha sweep outputs; out of scope. |
| `results/exp5_shd_*`, `results/exp8_*`, `results/exp9_*`, `results/exp10*`, `results/lif_*` | Spiking/SHD/LIF/ALIF outputs; out of scope. |
| `results/exp7_*`, `results/smnist_*` | sMNIST/psMNIST outputs; out of scope. |
| `experiments/depth_sweep.py` | Useful later only after depth-2 gates; currently over-emphasizes depth sweeps and includes depths 4-5, which are out of current scope. |

## Current Canonical Files

- `project_plan.md`
- `README.md`
- `PLAN_ALIGNMENT.md`
- `models/deep_rnn.py`
- `models/vanilla_rnn.py`
- `tasks/store_and_recall.py`
- `tasks/cue_accumulation.py`
- `learning_rules/bptt.py`
- `learning_rules/eprop.py`
- `learning_rules/deep_eprop.py`
- `learning_rules/deep_rtrl.py`
- `tests/sanity_checks.py`
- `experiments/single_layer_eprop.py` after a light cleanup/rerun
- `experiments/deep_eprop_comparison.py` after tightening to G2/G3 and delay
  sweep outputs
- `experiments/store_recall_suite.py`
- `notebooks/store_and_recall_colab.ipynb`
