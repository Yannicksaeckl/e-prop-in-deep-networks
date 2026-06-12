# Deep E-prop: Online Credit Assignment Across Depth in Recurrent Networks

**Project 5 — NeuroAI & ML 4 Neuro, Sommersemester 2026**
Yannick Säckl, Ruchit Kumar Patel, Simon Peter

---

## 0. TL;DR (the one result we are aiming for)

Take a **single** delay task (store-and-recall) and an architecture where **inserting a feedforward hidden layer between the recurrent layer and the readout makes naive e-prop fail while BPTT still solves it** (the failure documented in Bellec et al. 2020, Supp. Fig. 8). Then show that **Millidge's deep e-prop recursion recovers the learning that naive e-prop loses**, approaching BPTT.

That is the whole project. Everything else is a stretch goal and is explicitly labelled as such.

This reframing is deliberate and is the answer to the supervisor's note that *"depth doesn't matter much on these tasks."* It does not need to. Deep e-prop's claim is **not** "depth improves accuracy" — it is "a local, online rule can assign credit across depth." So depth must be load-bearing **for the credit-assignment problem**, not for task difficulty. The inserted-layer construction makes naive e-prop break *by construction*, which is exactly the condition under which deep e-prop's hierarchical trace has something to prove.

---

## 1. Research question

At a fixed architecture where credit must cross a non-recurrent hidden layer, can the deep e-prop recursion (Millidge, 2025) assign credit across depth well enough to match exact BPTT, and does its hierarchical eligibility trace recover learning that a naive (single-layer) e-prop baseline cannot?

Secondary, only if the core lands cleanly: as the delay grows, does the **temporal** trace still earn its keep (deep e-prop vs the `lambda = 0` / immediate-derivative ablation)?

---

## 2. Background (short)

- BPTT trains RNNs by storing all states and replaying backwards in time. Memory-heavy, biologically implausible.
- E-prop (Bellec et al., 2020) factorises the gradient into a forward-computable **eligibility trace** and a top-down **learning signal**, giving an online local rule. It approaches BPTT — but only in a **single** recurrent layer.
- Bellec et al. report the relevant failure mode: insert feedforward hidden layers between the recurrent net and the readout, and symmetric e-prop can no longer learn a task BPTT still solves. The inserted neurons block "route (i)" (the within-neuron slow-variable highway) and leave only the discarded "route (ii)" (the cross-neuron path).
- Millidge (2025) derives a recursion that propagates an eligibility trace **across depth as well as time**, yielding a per-layer trace. The note is purely mathematical — **no experiments**. This project supplies the first numerical test.

**Key references:** Millidge (2025), *Generalizing E-prop to Deep Networks* (arXiv:2512.24506); Bellec et al. (2020), *Nature Communications* 11:3625; Williams & Zipser (1989), RTRL. (Shalev-Merin 2026 and Zucchet et al. 2023 are relevant only to the optional `lambda = 0` ablation.)

---

## 3. Scope and non-goals (read this before adding anything)

These guardrails encode the supervisor feedback. Do not cross them without a deliberate decision.

**In scope (core):**
- One task: store-and-recall.
- Dense vanilla `tanh` RNN. No spiking.
- One architecture family: recurrent layer + 0/1/2 inserted feedforward hidden layers + readout.
- Four learning rules on the same model: BPTT, deep-RTRL (correctness gate only), deep e-prop, naive e-prop.
- Primary outcome: behavioural (accuracy / loss vs BPTT).

**Non-goals (do not implement unless the core is done and time remains):**
- ❌ A second task (no evidence accumulation). One task only.
- ❌ Spiking neurons (no LIF/ALIF, no `snntorch`, no traces-propagation port).
- ❌ Task-dependence as a variable.
- ❌ RTRL as a research result. We use it **only** as a numerical unit test (see §6).
- ❌ Treating gradient cosine similarity as the headline. It is secondary; the learning curve leads.
- ❌ Random / adaptive e-prop feedback variants. Use symmetric e-prop (`B = W_out^T`).

**Stretch (only if core lands):** delay-length sweep · `lambda = 0` temporal ablation · gradient cosine vs depth/time · depth sweep 1–3 · spectral-radius axis · a slow-timescale (non-spiking) adaptation channel.

---

## 4. Workflow (how this repo is run)

Human orchestrates; Coding agent implements; human sanity-checks at each gate and provides scientific reasoning.

- Each **milestone in §10** is a self-contained work unit handed to the coding agent.
- Every milestone has an **acceptance criterion** — a concrete, checkable output (a passing test, a printed number, a plot). Do not start milestone *n+1* until milestone *n*'s acceptance criterion is met and eyeballed.
- **Correctness gates (§6) are blocking.** If the deep-RTRL == BPTT gate fails, stop and fix the trace plumbing before touching e-prop. A wrong trace makes every downstream e-prop result uninterpretable.
- Prefer small, fast configs (tiny nets, short sequences, `float64`) for all gates so a full check runs in seconds on CPU.

---

## 5. Specifications (precise enough to implement)

### 5.1 Task: store-and-recall

A binary value must be stored on command and reproduced after a delay. The loss lands **only at the recall step**, which gives the sparse-learning-signal structure that makes temporal credit assignment non-trivial — this is intentional.

**Inputs** at each timestep `t` (sequence length `T`), a vector of 3 channels:
- `value_t` ∈ {0, 1} — the bit to (possibly) store.
- `store_t` ∈ {0, 1} — 1 at exactly one timestep `t_store`.
- `recall_t` ∈ {0, 1} — 1 at exactly one timestep `t_recall`, with `t_recall = t_store + D`.

**Target:** at `t_recall`, output the value that was present at `t_store`. No loss at other timesteps.

**Delay `D`** is the single complexity knob. Default `D = 20`. (Sweep is stretch.)

**Generation:** randomise `value` ~ Bernoulli(0.5); place `t_store` early; set `t_recall = t_store + D`; fill non-command steps with random distractor values so the net cannot cheat by integrating. Provide a `seed` argument; return `(inputs, targets, recall_mask)`.

**Loss:** cross-entropy at the masked recall step (1-bit → 2-way classification). Readout produces 2 logits.

> Default 1-bit. Widening to `k` bits later is a config change, not a rewrite.

### 5.2 Model

A stack of layers, **at most one of which is recurrent** (the bottom one); the inserted layer(s) are feedforward. This is the Bellec inserted-layer construction.

Recurrent layer (size `N`, default 64):
$$h_t = \tanh\!\big(W_{rec}\,h_{t-1} + W_{in}\,x_t + b_{rec}\big),\qquad h_0 = 0$$

Inserted feedforward hidden layer(s) (size `M`, default 64, count ∈ {0,1,2}); **no recurrence**:
$$g_t = \tanh\!\big(W_{ff}\,h_t + b_{ff}\big)$$

Readout (non-recurrent; use leak `kappa = 0` for the non-spiking case):
$$y_t = W_{out}\,g_t + b_{out}$$

Number of inserted layers is the variable that turns the naive-e-prop failure on and off:
- `n_ff = 0` → flat recurrent net; naive e-prop should work (reproduces Bellec single-layer result).
- `n_ff >= 1` → route (i) blocked; **naive e-prop expected to fail**, deep e-prop expected to recover.

### 5.3 Learning rules (all on the same model)

Notation follows Millidge (2025). Per layer `l`, hidden state `h^l_t`, inter-layer signal `z^l_t` (for a `tanh` layer `z^l_t = h^l_t`, so `∂z/∂h = I`).

**BPTT** — reference. Plain PyTorch autograd over the unrolled net. This is the target to approximate.

**Deep-RTRL** — *correctness gate only.* The same trace recursion as deep e-prop but using **total** derivatives. Mathematically equivalent to BPTT, so it must reproduce BPTT's gradient to numerical precision. Per-layer trace (Millidge Eq. 10):
$$\epsilon^l_t = \frac{\partial h^l_t}{\partial h^l_{t-1}}\,\epsilon^l_{t-1} + K,\qquad
K = \begin{cases}\dfrac{\partial h^l_t}{\partial h^{l-1}_t}\,\epsilon^{l-1}_t & \text{non-terminal layer}\\[2mm]\dfrac{\partial h^l_t}{\partial \theta} & \text{terminal (bottom) layer}\end{cases}$$
Loss gradient = sum over loss timesteps of the top-layer trace contracted with `dL/dy_t · dy_t/dh^L_t`.

**Deep e-prop** — *the method.* Identical recursion, but replace **total** derivatives with **partials** (drop indirect cross-state influences), and use the **symmetric e-prop learning signal** `L^t_j = ∂E/∂z^t_j` routed through `W_out` instead of the exact `dE/dz^t_j`. For two layers this is Millidge Eq. 9:
$$\epsilon^{z}_t = \frac{\partial z_t}{\partial h_t}\,\epsilon^{h}_t + \frac{\partial z_t}{\partial z_{t-1}}\,\epsilon^{z}_{t-1},\qquad
\epsilon^{h}_t = \frac{\partial h_t}{\partial \theta} + \frac{\partial h_t}{\partial h_{t-1}}\,\epsilon^{h}_{t-1}$$

**Naive e-prop** — *the baseline that should fail at depth.* Single-layer e-prop applied to the recurrent layer with the readout error routed back, **without** the hierarchical depth trace `ε^z` carrying credit through the inserted layer(s). Concretely: do not propagate the trace across depth — the recurrent layer's learning signal is whatever reaches it directly, which the inserted layer blocks.

**Ablations (stretch):**
- `lambda = 0` (immediate derivatives): zero out the temporal recursion `∂h_t/∂h_{t-1} · ε_{t-1}`. Isolates the **temporal** trace.
- depth-trace-dropped: zero the `ε^z` cross-layer term in deep e-prop. Isolates the **depth** trace. *(This is the ablation most on-point for the novel claim; prefer it over `lambda = 0` if only one fits.)*

---

## 6. Correctness gates (blocking, run on tiny `float64` configs)

| Gate | Check | Tolerance |
|------|-------|-----------|
| G1 — trace plumbing | deep-RTRL gradient vs autograd BPTT gradient, `N=4`, `M=4`, `T=10`, `n_ff=1` | max relative error `< 1e-5` |
| G2 — depth-1 reduction | deep e-prop with `n_ff=0` equals single-layer e-prop | exact (same code path) |
| G3 — e-prop ≈ BPTT direction (flat net) | deep e-prop vs BPTT gradient cosine on `n_ff=0` | cosine `> 0.9` (sanity, not exact) |
| G4 — failure precondition | naive e-prop does **not** solve the `n_ff>=1` task that BPTT solves | qualitative: clear accuracy gap |

**G1 is the most important line in this document.** Without it you cannot tell "deep e-prop's approximation is poor" from "the trace code is buggy." Get G1 green before interpreting any e-prop learning curve.

---

## 7. Metrics

**Primary (behavioural):** recall accuracy and loss vs training iterations, deep e-prop vs BPTT vs naive e-prop, on the `n_ff>=1` architecture. Averaged over ≥3 seeds with error bars; held-out test set.

**Secondary (approximation diagnostic):** gradient cosine similarity between deep e-prop and BPTT, resolved by layer and (if cheap) by time step. Reported alongside — **not as a proxy for** — the behavioural result.

**Tertiary (stretch):** iterations-to-threshold; robustness across seeds; delay sweep curves.

---

## 8. Minimal viable result vs full target

**MVR (the "great first result"):**
1. Store-and-recall implemented; BPTT solves it at `n_ff=1`.
2. G1 green (deep-RTRL == BPTT).
3. One learning-curve plot: deep e-prop vs naive e-prop vs BPTT at `n_ff=1`, showing deep e-prop recovers what naive e-prop loses.

**Full target (if time):** add seeds + error bars; add the depth-trace ablation; add a short delay sweep; add the layer-resolved gradient-cosine plot.

---

## 9. Suggested repo structure

```
deep-eprop/
├── PLAN.md                     # this file
├── README.md                   # short pointer to PLAN.md + how to run
├── requirements.txt            # torch, numpy, matplotlib  (no snntorch)
├── src/
│   ├── tasks/
│   │   └── store_and_recall.py # §5.1 generator + recall mask
│   ├── models/
│   │   └── stacked_rnn.py      # §5.2 recurrent + feedforward + readout
│   ├── learning/
│   │   ├── bptt.py             # autograd reference
│   │   ├── rtrl.py             # deep-RTRL (total derivatives) — gate only
│   │   ├── eprop.py            # deep e-prop + naive e-prop + ablations
│   │   └── traces.py           # shared per-layer trace recursion (Eq. 10)
│   ├── metrics.py              # gradient cosine, accuracy
│   └── train.py                # loop, seeding, logging
├── experiments/
│   ├── configs/                # yaml/json per run
│   └── run_core.py             # the MVR experiment
├── tests/
│   └── test_gradient_gates.py  # G1–G4
├── results/                    # plots, csv logs
└── notebooks/                  # sanity-checking scratch
```

Keep the trace recursion in **one** place (`traces.py`) parameterised by `derivative_mode ∈ {total, partial}` and `temporal ∈ {on, off}` and `depth ∈ {on, off}`. Deep-RTRL, deep e-prop, naive e-prop, and both ablations then differ only by flags — this is what makes G1→G2→deep e-prop nearly free and keeps the rules provably comparable.

---

## 10. Milestones (work units for Claude Code; human checks the acceptance criterion)

**M1 — Task + model scaffolding.**
Implement `store_and_recall.py` and `stacked_rnn.py` with configurable `N, M, n_ff, D, seed`.
*Acceptance:* a forward pass on a batch runs; the recall mask selects exactly one timestep; a quick BPTT training run drives recall accuracy on `n_ff=0` toward 1.0.

**M2 — BPTT baseline.**
`bptt.py` + `train.py` loop, logging loss/accuracy, seeding, held-out eval.
*Acceptance:* BPTT solves the task at `n_ff=1, D=20` (test accuracy clearly above chance, ideally ~1.0).

**M3 — Trace recursion + deep-RTRL (GATE G1).**
`traces.py` (Eq. 10) and `rtrl.py` (total-derivative mode).
*Acceptance:* **G1 passes** — deep-RTRL gradient matches autograd BPTT to `< 1e-5` on the tiny `float64` config. Blocking.

**M4 — Deep e-prop + naive e-prop (GATES G2, G3, G4).**
Flip `traces.py` to partial-derivative mode; add the symmetric learning signal; add the naive variant (no `ε^z`).
*Acceptance:* G2 exact; G3 cosine `> 0.9` on flat net; **G4 shows naive e-prop failing on `n_ff>=1` while BPTT succeeds.**

**M5 — Core comparison (the MVR plot).**
`run_core.py`: deep e-prop vs naive e-prop vs BPTT at `n_ff=1, D=20`.
*Acceptance:* a single figure where deep e-prop's curve recovers the naive-e-prop gap and tracks toward BPTT. **This is the deliverable.**

**M6 — Robustness (stretch).** ≥3 seeds, error bars, test split on all reported numbers.

**M7 — Diagnostics & ablation (stretch).** Layer-resolved gradient cosine; depth-trace-dropped ablation; short delay sweep `D ∈ {5,10,20,40}`.

---

## 11. Implementation notes / known gotchas

- **Make depth load-bearing for *credit*, not accuracy.** Do not rely on stacked *recurrent* layers to spontaneously break naive e-prop — on an easy task they often won't, and you'll get a null result. Use the **inserted feedforward layer** so the failure is structural and matches Bellec's documented construction.
- **`z = h` for `tanh` layers**, so `∂z/∂h = I`; the `ε^z` recursion simplifies but is still doing the cross-depth work — keep it explicit so the naive-vs-deep contrast is clean.
- **Symmetric e-prop only:** learning signal uses `W_out^T`. Skip random/adaptive feedback.
- **Sparse loss is the point.** Loss only at recall makes the learning signal near-zero for most steps; this is what stresses temporal credit assignment. Don't "fix" it by adding dense supervision.
- **Run gates in `float64`.** The 1e-5 tolerance in G1 assumes double precision and short sequences.
- **Per-parameter-group traces.** Separate `W_rec`, `W_in`, `W_ff` need their own traces (Millidge Discussion, limitation 1). Budget for this in `traces.py`.
- **e-prop accumulates, then updates.** Accumulate `L^t_j · e^t_{ji}` across the trial and apply the weight update at trial end (matches Bellec / Millidge limitation 3). Online-per-step updates make traces off-policy — avoid for now.

---

## 12. Robustness and controls (when reporting)

- Multiple seeds with error bars on every reported metric.
- Held-out test split for all tasks.
- Every e-prop number reported against exact BPTT; deep-RTRL as the implementation gate; naive e-prop as the lower-fidelity reference.
- If the `lambda` ablation is run, retune the learning rate per setting (the optimum shifts with decay).

---

## 13. Open decisions (flag for supervisor if they bite)

- 1-bit vs k-bit store-and-recall (default 1-bit).
- Whether `n_ff=1` reliably breaks naive e-prop at `D=20`, or whether a second inserted layer / longer delay is needed to make the gap unambiguous (decide empirically at M4/G4).
- Whether to report the depth-trace ablation (most on-point for the novel claim) instead of / alongside `lambda=0`.
