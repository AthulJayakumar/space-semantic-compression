"""Independent label overlap uses conventional empty-mask handling."""

import numpy as np
import pytest

from evaluation.label_fidelity import binary_overlap


def test_overlap_and_empty_masks() -> None:
    truth = np.array([[1, 0], [1, 0]])
    predicted = np.array([[1, 1], [0, 0]])
    assert binary_overlap(predicted, truth) == (0.5, 1 / 3)
    assert binary_overlap(np.zeros((2, 2)), np.zeros((2, 2))) == (1.0, 1.0)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        binary_overlap(np.zeros((2, 2)), np.zeros((2, 3)))
