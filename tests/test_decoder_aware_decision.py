"""The advancement rule uses geographic clusters and matched budgets."""

import pytest

from evaluation.decoder_aware_decision import advancement_decision, paired_group_comparison


def record(sample: str, group: str, method: str, value: float, budget: int = 500) -> dict[str, str]:
    return {
        "sample_id": sample,
        "geographic_group": group,
        "method": method,
        "budget_index": "0",
        "target_bytes": str(budget),
        "sus": str(value),
        "detector_retention": str(value / 100),
    }


def test_comparison_counts_groups_not_correlated_chips() -> None:
    rows = []
    for index in range(3):
        for chip in range(2):
            sample = f"{index}-{chip}"
            rows.extend((record(sample, f"group-{index}", "vqvae_random", 50), record(sample, f"group-{index}", "vqvae_decoder_aware", 60 + index)))
    result = paired_group_comparison(rows, "vqvae_decoder_aware", "vqvae_random", "sus")
    assert result["n_groups"] == 3
    assert result["mean_difference"] == 11


def test_unmatched_budgets_rejected() -> None:
    rows = [record("a", "a", "vqvae_random", 50), record("a", "a", "vqvae_decoder_aware", 60, 600)]
    with pytest.raises(ValueError, match="same per-image byte budget"):
        paired_group_comparison(rows, "vqvae_decoder_aware", "vqvae_random", "sus")


def test_candidate_with_no_gain_does_not_advance() -> None:
    rows = []
    for index in range(3):
        for method in ("vqvae_utility", "vqvae_random", "vqvae_entropy", "vqvae_decoder_impact", "vqvae_decoder_aware"):
            rows.append(record(str(index), f"group-{index}", method, 50 + index))
    assert advancement_decision(rows)["decision"] == "no_go"


def test_unpaired_chips_within_group_rejected() -> None:
    rows = [
        record("a", "group-1", "vqvae_random", 50),
        record("b", "group-1", "vqvae_decoder_aware", 60),
        record("c", "group-2", "vqvae_random", 50),
        record("c", "group-2", "vqvae_decoder_aware", 60),
    ]
    with pytest.raises(ValueError, match="Unpaired samples"):
        paired_group_comparison(rows, "vqvae_decoder_aware", "vqvae_random", "sus")
