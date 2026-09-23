"""Check event-level exclusion and one-image-per-event training selection."""

from __future__ import annotations

from collections import Counter

from scripts.prepare_ecofirebias_training_expansion import select_training_rows


def test_selection_is_balanced_and_excludes_reserved_events() -> None:
    continents = ("Africa", "Asia", "Europe", "North America", "Oceania", "South America")
    rows = [
        {
            "event_id": f"{continent}-{event}",
            "example_id": f"{continent}-{event}-{kind}",
            "split": "train",
            "kind": kind,
            "continent": continent,
        }
        for continent in continents
        for event in range(3)
        for kind in ("burn", "neg")
    ]
    excluded = {f"{continent}-0" for continent in continents}
    chosen = select_training_rows(rows, excluded, seed=42, events_per_continent=2)
    assert len(chosen) == 12
    assert len({row["event_id"] for row in chosen}) == 12
    assert not ({row["event_id"] for row in chosen} & excluded)
    assert Counter((row["continent"], row["kind"]) for row in chosen) == {
        (continent, kind): 1 for continent in continents for kind in ("burn", "neg")
    }


def test_selection_rejects_insufficient_unseen_events() -> None:
    rows = [{"event_id": "only", "example_id": "only-burn", "split": "train", "kind": "burn", "continent": "Africa"}]
    try:
        select_training_rows(rows, set(), seed=42, events_per_continent=2)
    except ValueError as error:
        assert "Insufficient" in str(error)
    else:
        raise AssertionError("Expected a data-integrity failure")
