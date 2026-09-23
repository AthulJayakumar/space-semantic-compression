"""Small deterministic checks for checkpoint-lineage replay."""

from __future__ import annotations

import numpy as np

from scripts.audit_checkpoint_event_lineage import training_indices


def test_training_indices_match_recorded_random_split() -> None:
    count = 439
    seed = 20260925
    indices = np.arange(count)
    np.random.default_rng(seed).shuffle(indices)
    expected = set(indices[88:].tolist())
    assert training_indices(count, 0.2, seed) == expected
    assert len(expected) == 351


def test_training_indices_empty() -> None:
    assert training_indices(0, 0.2, 1234) == set()
