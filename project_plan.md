# Project_Plan

---

## 1. Objective

Provide the first numerical test of Millidge's (2025) deep e-prop recursion: a local, online,
forward-computed rule that assigns credit across **both time and depth** via eligibility traces.
The reference is exact BPTT at the **same depth**.

**What "success" means.** Depth is treated as a *credit-assignment stressor for an online local
rule*, not as task capacity. We do **not** require depth to improve task accuracy (a single
recurrent layer may already solve the tasks; that is fine). Success is:

1. deep e-prop trained at depth `D` reaches the loss/accuracy that BPTT reaches at depth `D`, and
2. the per-layer gradient cosine to BPTT stays high **including the bottom layer**.

Both are measurable even when depth is unnecessary for the task: BPTT still defines a target
gradient for the bottom layer, and the question is whether the traces recover it.

---

## 2. Scope

**In:** stacked vanilla tanh-RNNs (depth 1–2, optionally 3); store-and-recall and cue accumulation;
BPTT, deep e-prop, a d=0 reference, and deep-RTRL as a correctness gate; gradient-cosine and
performance diagnostics; delay as the single swept variable.

**Out (non-goals):** spiking / ALIF neurons; reward-based e-prop / actor–critic; the
feedforward-insertion architecture (see §3 collapse note); the full multi-rung rule ladder beyond
`{BPTT, deep-RTRL gate, deep e-prop, d=0}`; spectral-radius sweeps; any "depth improves accuracy"
(depth-as-capacity) claim. These can be revisited only after the core result lands.

---

## 3. Model

Stacked vanilla tanh-RNN. **All stacked layers are recurrent.** Let `D^l_t = diag(1 - (h^l_t)^2)`.

```
h^1_t = tanh(W^1_rec h^1_{t-1} + W^1_in x_t   + b^1)          # bottom layer
h^l_t = tanh(W^l_rec h^l_{t-1} + W^l_in h^{l-1}_t + b^l)      # l > 1
y_t   = W_out h^L_t + b_out                                   # readout from top layer L
```

Jacobians used by every trace-based rule:

- temporal (within layer): `∂h^l_t/∂h^l_{t-1} = D^l_t W^l_rec`
- depth (same timestep):   `∂h^l_t/∂h^{l-1}_t = D^l_t W^l_in`

**Collapse note (critical).** Do **not** insert a non-recurrent (feedforward) layer between the
recurrent layer and readout. With a non-recurrent upper layer `∂z_t/∂z_{t-1} = 0`, Millidge's
depth recursion collapses to per-layer single-layer e-prop and there is nothing novel to test.
Both stacked layers must carry recurrence.

---

## 4. Learning rules

All trace-based rules share one flag-parameterised trace module. Per parameter group, the
eligibility trace follows Millidge's recursion (Eq. 10), terminal at the group's own layer:

```
ε^l_t = (∂h^l_t/∂h^l_{t-1}) · ε^l_{t-1} + K
K = (∂h^l_t/∂h^{l-1}_t) · ε^{l-1}_t      # non-terminal (a layer below exists)
K = ∂h^l_t/∂θ                            # terminal (parameter group's own layer)
```

The loss gradient sums, over timesteps with a loss, the top-layer trace times the learning signal.
Each parameter group needs its own trace set (Millidge limitation #1); the agent handles this.

Variants differ only in **which derivatives enter the recursion**:

| Rule | Temporal term | Depth term | Role |
|---|---|---|---|
| **BPTT** | — (autograd) | — (autograd) | reference / target |
| **deep-RTRL** | full Jacobian (total derivs) | full | **correctness gate** — equals BPTT |
| **deep e-prop** | local / per-neuron partial (Bellec Eq. 13–14) | kept | object of study |
| **d=0** | dropped (`ε_{t-1}` term → 0) | kept | cheap lower-fidelity reference |

- The e-prop approximation is `dh/dθ ≈ ∂h/∂θ`: it ignores indirect influences mediated by other
  neurons' states across time, keeping the local temporal term and the instantaneous depth term.
- **d=0** keeps instantaneous depth + parameter terms but no temporal carry; it still does spatial
  credit across depth, so it is the natural foil for the delay sweep (§6).
- Learning signal uses the partial `∂E/∂z` (online), not the total `dE/dz`.

**Gate logic.** deep-RTRL must match BPTT to numerical precision — that validates the trace
plumbing. deep e-prop is *expected* to diverge from BPTT; that divergence is the result, **not** a
bug, so deep e-prop is never gated against BPTT.

---

## 5. Tasks

Both tasks use a three-phase trial so that **delay** is a clean, isolated knob.

### Store-and-recall
`encode (t_enc steps: present value) → delay (D steps: null/distractor) → recall (go-cue: output
the stored value for t_out steps)`. Loss computed during **recall only**.

### Cue accumulation (evidence integration, after Bellec)
`cues (left/right cues over time) → delay (D steps: no cues) → decision (output which side had more
cues)`. Loss computed during **decision only**.

**Delay length `D` = number of timesteps in the delay phase.** Hold `t_enc`, `t_out`, cue count,
and all other phase lengths fixed; vary only `D`. This isolates memory duration from
encoding/decoding difficulty and total input length.

Because the loss is concentrated at recall/decision, the learning signal is zero for most
timesteps **by design** — bridging that gap is the entire job of the eligibility trace, and it is
why d=0 should fail as `D` grows.

---

## 6. Experiments

Minimal, per task, at fixed depth `D_net = 2` (depth 1 = Bellec reproduction; depth 3 only if time):

1. **Core comparison:** BPTT vs deep e-prop — performance and per-layer/per-step gradient cosine.
2. **Delay sweep (single variable):** `{BPTT, deep e-prop, d=0}` across increasing `D`. Expected
   ordering as `D` grows: d=0 degrades first, deep e-prop holds longer, BPTT holds longest.

Run on store-and-recall first; repeat the core comparison and delay sweep on cue accumulation.

---

## 7. Metrics

**Primary**
- Final loss / accuracy relative to BPTT at the same depth.
- Gradient cosine similarity between deep e-prop and exact BPTT, **resolved by layer and by step**,
  with the bottom layer as the headline. Treated as a mechanistic indicator of the approximation,
  reported alongside (not as a proxy for) the behavioural outcome.

**Secondary**
- Iterations-to-threshold (steps to a fixed recall/decision accuracy) with a convergence-based
  stop, so a plateaued tail neither wastes compute nor distorts the metric.
- Seed robustness (error bars) on the **headline comparison only**.

---

## 8. Correctness gates / milestones

| Gate | Deliverable | Acceptance criterion |
|---|---|---|
| **G0** | Stacked tanh-RNN forward pass + BPTT training | BPTT solves store-and-recall at depth 1 and 2 |
| **G1** | Single-layer e-prop | depth-1 e-prop tracks BPTT (reproduces Bellec) |
| **G2** | deep-RTRL numerical gate | on a toy net (`H≈16`, few steps), trace gradient matches BPTT to ≤ ~1e-6 (max abs/rel error) |
| **G3** | deep e-prop core result | depth-2 deep e-prop reaches BPTT loss within margin on store-and-recall; bottom-layer cosine above threshold |
| **G4** | Generalisation + sweep | core comparison reproduced on cue accumulation; delay sweep (`BPTT/deep e-prop/d=0`) with seeds + error bars |

G2 is the load-bearing correctness check: run RTRL **only** as this one-off gate (never as a
training run) to keep cost negligible.

**Minimal viable result:** G1 + G2 + a first depth-2 deep e-prop vs d=0 vs BPTT learning curve and
gradient-cosine plot.

---

## 9. Implementation notes

- **Notation.** Bellec uses `z` for observable spikes; Millidge uses `z` for the upper hidden
  layer. This doc uses `h^l` for the hidden state of layer `l` and has no spikes — disambiguate
  explicitly in code and comments.
- **Efficiency.** Small hidden sizes (e.g. `H = 64`) and short sequences; CPU / Colab is enough.
  RTRL (`O(H^4)`/step) appears only in the G2 toy gate. Shared trace module, flag-selected rule.
- **`λ` / learning rate.** `λ=0` (the d=0 limit) is a first-class reference, not a fixed constant.
  Retune the learning rate per depth and per decay (Shalev-Merin: `λ` is largely
  learning-rate-compensable, and the optimum shifts with both).
- **References.** Millidge (2025, arXiv:2512.24506); Bellec et al. (2020, Nat. Commun. 11:3625).
  Reuse `github.com/IGITUGraz/eligibility_propagation` for single-layer sanity checks and
  `github.com/NicolasZucchet/Online-learning-LR-dependencies` for the multi-layer online setup.
