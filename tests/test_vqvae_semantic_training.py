"""Tests for the mission-consistency VQ-VAE training objective."""

from __future__ import annotations

import torch

from scripts.finetune_vqvae_satellite import prune_tokens_by_utility, semantic_consistency_loss
from semantic_ai.burn_scar_model import BurnScarUtilityNet
from src.models.vqvae import VectorQuantizerEMA


def test_semantic_consistency_loss_backpropagates_to_reconstruction_only():
    detector = BurnScarUtilityNet(base_channels=8).eval()
    for parameter in detector.parameters():
        parameter.requires_grad = False
    target = torch.rand((1, 3, 32, 32)) * 2.0 - 1.0
    reconstruction = target.detach().clone().requires_grad_(True)
    loss = semantic_consistency_loss(reconstruction, target, detector, 1.0, 2.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert reconstruction.grad is not None
    assert torch.isfinite(reconstruction.grad).all()
    assert all(parameter.grad is None for parameter in detector.parameters())


def test_commitment_loss_updates_encoder_representation():
    quantizer = VectorQuantizerEMA(n_codes=16, d=8).eval()
    encoded = torch.randn((1, 8, 4, 4), requires_grad=True)
    _, _, commitment = quantizer(encoded)
    commitment.backward()
    assert encoded.grad is not None
    assert float(encoded.grad.abs().sum()) > 0.0


def test_prune_tokens_by_utility_keeps_highest_utility_codes():
    tokens = torch.tensor([[[1, 1], [2, 3]]])
    utility = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]]]])

    pruned = prune_tokens_by_utility(tokens, utility, keep_ratio=0.5)

    assert pruned.tolist() == [[[1, 1], [2, 1]]]


def test_wire_fallback_uses_only_selected_codes():
    tokens = torch.tensor([[[1, 1], [3, 4]]])
    utility = torch.tensor([[[[0.1, 0.2], [0.9, 0.8]]]])
    pruned = prune_tokens_by_utility(tokens, utility, keep_ratio=0.5, wire_fallback=True)
    assert pruned.tolist() == [[[3, 3], [3, 4]]]
