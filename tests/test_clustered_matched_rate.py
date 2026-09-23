"""Checks that related image chips count as one independent wildfire event."""

import pytest

from evaluation.clustered_matched_rate import analyze, compare_checkpoints, event_means


def row(sample: str, event: str, method: str, sus: float) -> dict[str, str]:
    return {
        "sample_id": sample,
        "event_group": event,
        "method": method,
        "budget_index": "0",
        "sus": str(sus),
        "detector_retention": "0.8",
        "psnr": "20",
        "ssim": "0.7",
        "lpips": "0.3",
    }


def test_event_means_average_correlated_chips() -> None:
    rows = [row("a", "event-1", "jpeg", 10), row("b", "event-1", "jpeg", 30)]
    assert event_means(rows)[("jpeg", 0, "event-1")]["sus"] == 20


def test_paired_tests_count_events_not_chips() -> None:
    rows = []
    for index in range(3):
        for chip in range(2):
            sample = f"{index}-{chip}"
            rows.extend((row(sample, f"event-{index}", "jpeg", 50), row(sample, f"event-{index}", "vqvae_utility", 60 + index)))
    result = next(item for item in analyze(rows) if item["metric"] == "sus")
    assert result["n_events"] == 3
    assert result["mean_difference"] == 11
    assert result["paired_t_p_holm"] >= result["paired_t_p"]


def test_missing_event_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="event_group"):
        event_means([row("a", "unknown", "jpeg", 20)])


def test_checkpoint_comparison_requires_complete_matched_byte_budget() -> None:
    baseline, candidate = [], []
    for index in range(3):
        old = row(str(index), f"event-{index}", "vqvae_utility", 20)
        new = row(str(index), f"event-{index}", "vqvae_utility", 30 + index)
        old["target_bytes"] = "500"
        new["target_bytes"] = "500" if index < 2 else "600"
        baseline.append(old)
        candidate.append(new)
    assert compare_checkpoints(baseline, candidate) == []
    candidate[-1]["target_bytes"] = "500"
    result = next(item for item in compare_checkpoints(baseline, candidate) if item["metric"] == "sus")
    assert result["n_events"] == 3
    assert result["n_matched_chips"] == 3
