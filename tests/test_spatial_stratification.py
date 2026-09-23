"""Spatial ordering gives every block an early opportunity to transmit."""

import numpy as np
import pytest

from token_selection.spatial_stratification import stratified_order


def test_first_layer_covers_all_blocks_and_order_is_permutation() -> None:
    scores = np.arange(16, dtype="float32").reshape(4, 4)
    order = stratified_order(scores, block_size=2)
    assert sorted(order.tolist()) == list(range(16))
    assert {int(position // 4 // 2 * 2 + position % 4 // 2) for position in order[:4]} == {0, 1, 2, 3}


def test_invalid_score_shape_rejected() -> None:
    with pytest.raises(ValueError, match="2D"):
        stratified_order(np.ones(4), block_size=2)
