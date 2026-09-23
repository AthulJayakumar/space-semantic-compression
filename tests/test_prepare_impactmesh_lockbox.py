from __future__ import annotations

import numpy as np

from scripts.prepare_impactmesh_lockbox import event_balanced_selection, normalize_rgb


def test_event_balanced_selection_is_deterministic_and_spans_events() -> None:
    sample_ids = [
        "EMSR001_1_TILE_A",
        "EMSR001_1_TILE_B",
        "EMSR001_1_TILE_C",
        "EMSR002_1_TILE_A",
        "EMSR002_1_TILE_B",
        "EMSR003_1_TILE_A",
    ]

    first = event_balanced_selection(sample_ids, count=4, seed=7)
    second = event_balanced_selection(sample_ids, count=4, seed=7)

    assert first == second
    assert len(first) == len(set(first)) == 4
    assert {"_".join(value.split("_")[:2]) for value in first} == {
        "EMSR001_1",
        "EMSR002_1",
        "EMSR003_1",
    }


def test_normalize_rgb_accepts_channel_first_and_returns_uint8() -> None:
    channel_first = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    normalized = normalize_rgb(channel_first)

    assert normalized.shape == (4, 5, 3)
    assert normalized.dtype == np.uint8
