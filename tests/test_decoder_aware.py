"""Decoder-impact ranking follows the same fallback rule as token pruning."""

import numpy as np
import pytest
import torch

from token_selection.decoder_aware import blended_decoder_aware_scores, decoder_impact_scores


def test_decoder_impact_uses_modal_fallback_code() -> None:
    tokens = torch.tensor([[[0, 0], [1, 2]]])
    embedding = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    scores = decoder_impact_scores(tokens, embedding)
    assert scores[0, 0] == 0
    assert scores[1, 1] == 1
    assert 0 < scores[1, 0] < 1


def test_decoder_aware_blend_validates_shape() -> None:
    assert blended_decoder_aware_scores(np.ones((2, 2)), np.zeros((2, 2))).mean() == 0.5
    with pytest.raises(ValueError, match="token grid"):
        blended_decoder_aware_scores(np.ones((2, 2)), np.ones((2, 3)))
