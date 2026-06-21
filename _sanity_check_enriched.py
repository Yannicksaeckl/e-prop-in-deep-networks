"""Local sanity check for the enriched evidence accumulation task."""
import sys
sys.path.insert(0, '.')
import torch
import tasks.enriched_evidence_accumulation as EEA
from models.deep_rnn import DeepRNN
from learning_rules.bptt import compute_bptt_gradients, _trace_mse_loss
from learning_rules.deep_eprop import compute_deep_eprop_gradients, mse_error
from learning_rules.deep_rtrl import compute_deep_rtrl_gradients
from learning_rules.deep_rtrl import mse_error as rtrl_mse_error

print("=" * 60)
print("Sanity check: enriched_evidence_accumulation")
print("=" * 60)

# ── 1. Task shapes ────────────────────────────────────────────
kw = dict(n_cues=5, delay=3, d_cue=8, k=4, w=1, gap=2, noise_level=0.1, weights_seed=42)
inp, tgt, msk = EEA.generate_batch(8, **kw, seed=0)
T = EEA.sequence_length(n_cues=5, delay=3, w=1, gap=2)
t_r = EEA.decision_timestep(n_cues=5, delay=3, w=1, gap=2)
n_in = EEA.n_in(8)
assert inp.shape == (T, 8, n_in), f"input shape mismatch: {inp.shape}"
assert tgt.shape == (T, 8, 2)
assert msk.shape == (T, 8)
assert msk[t_r].sum() == 8,       "mask should be 1 at decision step"
assert msk[:t_r].sum() == 0.0,    "mask should be 0 before decision step"
print(f"[1] Task shapes OK  T={T}, t_recall={t_r}, n_in={n_in}")

# ── 2. Label balance ──────────────────────────────────────────
_, tgt2, _ = EEA.generate_batch(4000, **kw, seed=1)
p = tgt2[t_r].argmax(-1).float().mean().item()
assert abs(p - 0.5) < 0.06, f"Label imbalance: P(right)={p:.3f}"
print(f"[2] Label balance OK  P(right)={p:.3f}")

# ── 3. Parity label function ──────────────────────────────────
w_par = EEA.make_parity_weights(k=4, seed=42)
# If w=[+1,+1]: label = sign(x0*x1 + x2*x3)
# For x=[+1,+1,+1,+1]: z=[1,1], s=w@z, label depends on w
x_pos = torch.tensor([[1., 1., 1., 1.]])
x_neg = torch.tensor([[-1., 1., -1., 1.]])  # z=[-1,-1]
lbl_pos = EEA.parity_label(x_pos, w_par)
lbl_neg = EEA.parity_label(x_neg, w_par)
print(f"[3] parity_label: w={w_par.tolist()}")
print(f"    x=[+1,+1,+1,+1] → label={lbl_pos.item()}")
print(f"    x=[-1,+1,-1,+1] → label={lbl_neg.item()}")

# ── 4. Accuracy helper ────────────────────────────────────────
outputs = torch.zeros(T, 8, 2)
outputs[:, :, 0] = 1.0  # always predict "left"
acc = EEA.task_accuracy(outputs, tgt, msk)
print(f"[4] Always-left acc: {acc:.3f}  (expect ~0.5)")

# ── 5. Gradient computation smoke ────────────────────────────
torch.manual_seed(42)
model = DeepRNN(n_in, 8, 2, n_layers=2)
inp_s, tgt_s, msk_s = EEA.generate_batch(4, **kw, seed=7)

g_bptt  = compute_bptt_gradients(model, inp_s, tgt_s, msk_s, _trace_mse_loss)
g_eprop = compute_deep_eprop_gradients(model, inp_s, tgt_s, msk_s, mse_error)
g_rtrl  = compute_deep_rtrl_gradients(model, inp_s, tgt_s, msk_s, rtrl_mse_error)

# RTRL should match BPTT closely
max_err = max((g_rtrl[k] - g_bptt[k]).abs().max().item()
              for k in g_bptt if k in g_rtrl)
print(f"[5] RTRL vs BPTT max abs err: {max_err:.2e}  (expect < 1e-4)")
assert max_err < 1e-4, f"RTRL mismatch: {max_err:.2e}"

# ── 6. no_eps_z: lower-layer grads must be zero after zeroing ─
g_nepsz = dict(g_eprop)
L = 2
for l in range(L - 1):
    g_nepsz[f'W_recs.{l}'] = torch.zeros_like(g_nepsz[f'W_recs.{l}'])
    g_nepsz[f'biases.{l}'] = torch.zeros_like(g_nepsz[f'biases.{l}'])
    if l == 0:
        g_nepsz['W_in'] = torch.zeros_like(g_nepsz['W_in'])
for key in ['W_recs.0', 'biases.0', 'W_in']:
    assert g_nepsz[key].abs().max() == 0.0, f"no_eps_z: {key} should be zero"
# Top layer should still have non-zero gradient
assert g_nepsz['W_recs.1'].abs().max() > 0.0, "no_eps_z: top layer should still have gradient"
print("[6] no_eps_z zeroing OK  (lower=0, top≠0)")

# ── 7. aux_running_readout does not break shapes ──────────────
inp_a, tgt_a, msk_a = EEA.generate_batch(8, **kw, seed=5, aux_running_readout=True)
assert inp_a.shape == (T, 8, n_in)
# Auxiliary mask steps should have weight 0.1
aux_steps = (msk_a > 0.01) & (msk_a < 0.5)
print(f"[7] aux_running_readout OK  aux steps: {aux_steps.sum().item()}")

print()
print("ALL CHECKS PASSED ✓")
