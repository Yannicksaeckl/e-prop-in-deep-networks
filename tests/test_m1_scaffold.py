import torch

from models.stacked_rnn import StackedRNN
from tasks.store_and_recall import generate_batch, masked_cross_entropy, task_accuracy


def test_store_and_recall_shapes_and_seeded_values():
    inputs_a, targets_a, mask_a = generate_batch(8, delay=5, seed=123)
    inputs_b, targets_b, mask_b = generate_batch(8, delay=5, seed=123)

    assert inputs_a.shape == (7, 8, 3)
    assert targets_a.shape == (7, 8)
    assert mask_a.shape == (7, 8)
    assert torch.equal(inputs_a, inputs_b)
    assert torch.equal(targets_a, targets_b)
    assert torch.equal(mask_a, mask_b)
    assert torch.all(mask_a.sum(dim=0) == 1)

    recall_t = mask_a[:, 0].nonzero(as_tuple=False).item()
    store_t = recall_t - 5
    stored_bits = inputs_a[store_t, :, 0].long()
    assert torch.equal(targets_a[recall_t], stored_bits)
    assert torch.all(inputs_a[store_t, :, 1] == 1)
    assert torch.all(inputs_a[recall_t, :, 2] == 1)


def test_stacked_rnn_forward_for_all_m1_depths():
    inputs, targets, mask = generate_batch(4, delay=3, seed=7)

    for n_ff in (0, 1, 2):
        model = StackedRNN(n_in=3, n_rec=6, n_out=2, n_ff=n_ff)
        logits, states = model(inputs)

        assert logits.shape == (5, 4, 2)
        assert len(states) == 6
        assert len(states[0]) == 1 + n_ff
        assert torch.isfinite(masked_cross_entropy(logits, targets, mask))
        assert 0.0 <= task_accuracy(logits, targets, mask) <= 1.0
