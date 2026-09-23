from __future__ import annotations

from scripts.prepare_ecofirebias_lockbox import balanced_event_pairs


def test_balanced_event_pairs_selects_one_burn_and_control_per_continent_event() -> None:
    rows = []
    for continent in ("Africa", "Europe"):
        for event in ("event-a", "event-b"):
            for kind in ("burn", "neg"):
                rows.append(
                    {
                        "split": "test",
                        "kind": kind,
                        "continent": continent,
                        "event_id": f"{continent}-{event}",
                        "example_id": f"{continent}-{event}-{kind}",
                    }
                )

    selected = balanced_event_pairs(rows, events_per_continent=1, seed=3)

    assert len(selected) == 4
    assert {row["continent"] for row in selected} == {"Africa", "Europe"}
    assert {row["kind"] for row in selected} == {"burn", "neg"}
    assert len({row["event_id"] for row in selected}) == 2


def test_balanced_event_pairs_excludes_prior_lockbox_events() -> None:
    rows = [
        {"split": "test", "kind": kind, "continent": "Africa", "event_id": event, "example_id": f"{event}-{kind}"}
        for event in ("used", "fresh") for kind in ("burn", "neg")
    ]
    selected = balanced_event_pairs(rows, events_per_continent=1, seed=3, excluded_events={"used"})
    assert {row["event_id"] for row in selected} == {"fresh"}
