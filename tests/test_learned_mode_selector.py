"""tests.test_learned_mode_selector

Plain-English purpose: Automated checks for the optional learned token-priority model.

These tests verify the model shape and mode-conditioning behavior without
requiring a VQ-VAE checkpoint or satellite dataset.
"""

import torch

from token_selection.learned_mode_selector import (
    ModeConditionedSelectorConfig,
    ModeConditionedTokenScorer,
    topk_mask,
)


def test_mode_conditioned_scorer_returns_token_grid_scores():
    model = ModeConditionedTokenScorer(ModeConditionedSelectorConfig(codebook_size=32, hidden_channels=8))
    tokens = torch.randint(0, 32, (2, 4, 4))
    utility = torch.rand(2, 4, 4)
    entropy = torch.rand(2, 4, 4)
    detail = torch.rand(2, 4, 4)

    scores = model.score(tokens, utility, entropy, detail, "mission_utility")

    assert scores.shape == (2, 4, 4)
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)


def test_mode_conditioning_changes_scores():
    torch.manual_seed(7)
    model = ModeConditionedTokenScorer(ModeConditionedSelectorConfig(codebook_size=32, hidden_channels=8))
    tokens = torch.randint(0, 32, (1, 4, 4))
    utility = torch.rand(1, 4, 4)
    entropy = torch.rand(1, 4, 4)
    detail = torch.rand(1, 4, 4)

    mission_scores = model.score(tokens, utility, entropy, detail, "mission_utility")
    reconstruction_scores = model.score(tokens, utility, entropy, detail, "reconstruction_balanced")

    assert not torch.allclose(mission_scores, reconstruction_scores)


def test_topk_mask_keeps_requested_number_of_tokens():
    scores = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)

    mask = topk_mask(scores, keep_ratio=0.25)

    assert mask.shape == (1, 4, 4)
    assert int(mask.sum()) == 4
    assert mask[0, 3, 3]
