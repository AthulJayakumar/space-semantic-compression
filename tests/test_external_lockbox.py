"""Tests for deterministic external lockbox pairing and selection."""

from __future__ import annotations

from evaluation.external_lockbox import pair_firescope_files, select_lockbox_candidates


def _paths() -> list[str]:
    paths: list[str] = []
    for region in ("europe", "usa"):
        for index in range(8):
            stem = f"tile_R{index % 4}_2020_lon{index}.0_lat{40 + index}.0"
            paths.extend(
                [
                    f"{region}/wildfire_events/images/2019/{stem}.png",
                    f"{region}/wildfire_events/masks/2020/{stem}.npy",
                ]
            )
        for index in range(4):
            stem = f"tile_NN_negative_lon{index}.0_lat{20 + index}.0"
            paths.extend(
                [
                    f"{region}/wildfire_events/controls_images/{stem}.png",
                    f"{region}/wildfire_events/controls/{stem}.npy",
                ]
            )
    return paths


def test_pair_firescope_files_matches_positive_and_control_masks():
    candidates = pair_firescope_files(_paths())
    assert len(candidates) == 24
    assert {candidate.fire_status for candidate in candidates} == {"positive", "negative"}
    assert {candidate.region for candidate in candidates} == {"europe", "usa"}


def test_lockbox_selection_is_deterministic_and_balanced():
    candidates = pair_firescope_files(_paths())
    quotas = {
        "europe_positive": 4,
        "usa_positive": 4,
        "europe_negative": 2,
        "usa_negative": 2,
    }
    first = select_lockbox_candidates(candidates, quotas=quotas, seed="test")
    second = select_lockbox_candidates(reversed(candidates), quotas=quotas, seed="test")
    assert first == second
    assert len(first) == 12
    assert sum(item.fire_status == "positive" for item in first) == 8
    assert sum(item.region == "europe" for item in first) == 6
