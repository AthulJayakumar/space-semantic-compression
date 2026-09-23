"""Fixed-budget gates have exact forward sparsity and trainable gradients."""

import pytest
import torch

from token_selection.task_aligned import decode_gated_tokens, straight_through_topk


def test_straight_through_gate_is_exact_in_forward_and_differentiable() -> None:
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).requires_grad_()
    gate = straight_through_topk(logits, 0.25, 0.2)
    assert int(gate.sum().item()) == 4
    assert torch.allclose(gate, (gate > 0.5).float(), atol=1e-6)
    (gate * torch.arange(16).reshape(1, 4, 4)).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_decode_rejects_mismatched_grids() -> None:
    with pytest.raises(ValueError, match="share"):
        decode_gated_tokens(torch.nn.Linear(1, 1), torch.zeros(1, 2, 2, dtype=torch.long), torch.zeros(1, 3, 3))
