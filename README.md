# Deep E-prop In Deep Recurrent Networks

This repository is now aligned to `project_plan.md`, the supervisor-updated
plan for a narrow test of Millidge's deep e-prop recursion in stacked vanilla
tanh RNNs.

The current core question is not whether depth improves accuracy. Depth is a
credit-assignment stressor: at the same network depth, does deep e-prop reach
BPTT-like performance, and does its gradient stay aligned with BPTT, especially
in the bottom recurrent layer?

## Current Scope

In scope:

- stacked vanilla tanh RNNs, depth 1-2, with depth 3 optional later
- store-and-recall first, then cue accumulation
- BPTT, deep-RTRL as a numerical gate, deep e-prop, and d=0
- delay as the primary swept variable
- final performance plus layer/step-resolved gradient cosine diagnostics

Out of scope for the core result:

- spiking, LIF, ALIF, and SHD experiments
- leaky-RNN alpha sweeps
- sMNIST/psMNIST long-sequence experiments
- feedforward insertion architectures
- spectral-radius sweeps
- depth-as-capacity claims

See `PLAN_ALIGNMENT.md` for a concrete done/not-done checklist and redundant
file inventory.

## Repository Structure

```text
tasks/           benchmark tasks: store-and-recall, cue accumulation
models/          core DeepRNN/VanillaRNN plus legacy out-of-scope models
learning_rules/  BPTT, single-layer e-prop, deep e-prop/d=0, deep-RTRL
experiments/     scripts for the sequential experiment path
tests/           fast sanity checks for task shapes and numerical gates
results/         generated figures/metrics, including legacy out-of-scope runs
figures/         generated diagrams
```

## Sequential Milestones

| Gate | Status | Current repo state |
|---|---:|---|
| G0: stacked tanh-RNN + BPTT training | Partial | forward pass and BPTT gradients exist; full solve-at-depth-1-and-2 run still needs a clean gated script/result |
| G1: single-layer e-prop | Partial | implementation and depth-1 equivalence check exist; training reproduction needs a current rerun |
| G2: deep-RTRL numerical gate | Implemented | `tests/sanity_checks.py` checks absolute/relative agreement with matching BPTT loss normalization |
| G3: depth-2 deep e-prop core result | Partial | deep e-prop and d=0 implementation exists; store-and-recall learning/cosine result needs a fresh plan-aligned run |
| G4: cue accumulation + delay sweep | Not yet | task exists; plan-aligned BPTT/deep-eprop/d=0 sweep with seeds/error bars is still future work |

## Quick Check

```bash
python tests/sanity_checks.py
```

The sanity suite is a plumbing check. Passing it does not mean the training
milestones have been fully accepted.
