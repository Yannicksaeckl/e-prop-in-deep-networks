# Deep E-prop: Online Credit Assignment Across Depth

**NeuroAI & ML 4 Neuro - Sommersemester 2026**

Group: Simon Peter, Yannick Saeckl, Ruchit Kumar Patel

## Project Overview

This repo tests whether Millidge's deep e-prop recursion can assign credit
across inserted feedforward hidden layers in a recurrent network. The core
experiment is the 1-bit store-and-recall task from `PLAN.md`: BPTT should solve
the task, naive e-prop should fail once a feedforward hidden layer blocks the
direct recurrent-to-readout path, and deep e-prop should recover the missing
credit assignment.

## Current Layout

```
tasks/           # Store-and-recall generator and task metrics
models/          # RNN model definitions
learning_rules/  # BPTT, e-prop, deep e-prop, deep-RTRL
experiments/     # Experiment scripts
results/         # Output figures and metrics
tests/           # Fast correctness/smoke checks
```

M1 uses the existing root-level package layout. New code should use
`tasks.store_and_recall` and `models.stacked_rnn`. Older files remain available
for reference but still reflect the previous project framing.

## Colab Workflow

Use GitHub as the source of truth and run repo scripts/tests from a fresh clone:

```python
!git clone -b <branch> <repo-url>
%cd e-prop-in-deep-networks
!pip install -r requirements.txt
!pytest -q
```

Use CPU for tiny correctness gates and smoke checks. Use GPU only for longer
learning curves. Save generated figures/logs under `results/`, then commit only
the selected artifacts that belong in the report.

## Key References

- Bellec et al. (2020) - E-prop: Biologically plausible learning in RNNs
- Millidge (2025) - Deep E-prop
- Williams & Zipser (1989) - RTRL
