"""Geography-clustered advancement decision for decoder-aware selectors."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from evaluation.clustered_matched_rate import bootstrap_ci


CANDIDATES = ("vqvae_decoder_impact", "vqvae_decoder_aware")
CONTROLS = ("vqvae_utility", "vqvae_random", "vqvae_entropy")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_group_comparison(
    rows: list[dict[str, str]], candidate: str, baseline: str, metric: str, budget_index: int = 0
) -> dict[str, object]:
    per_group: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    per_sample_budget: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if int(row["budget_index"]) != budget_index or row["method"] not in (candidate, baseline):
            continue
        group = row.get("geographic_group", "").strip()
        if not group or group == "unknown":
            raise ValueError("Geographic group is required for clustered inference")
        sample_values = per_group[(row["method"], group)]
        if row["sample_id"] in sample_values:
            raise ValueError("Duplicate sample in matched-rate results")
        sample_values[row["sample_id"]] = float(row[metric])
        per_sample_budget[row["sample_id"]].add(int(row["target_bytes"]))
    if any(len(budgets) != 1 for budgets in per_sample_budget.values()):
        raise ValueError("Methods did not receive the same per-image byte budget")
    groups = sorted(
        group for method, group in per_group
        if method == candidate and (baseline, group) in per_group
    )
    if len(groups) < 2:
        raise ValueError("At least two matched geographic groups are needed")
    differences_by_group = []
    for group in groups:
        candidate_samples = per_group[(candidate, group)]
        baseline_samples = per_group[(baseline, group)]
        if candidate_samples.keys() != baseline_samples.keys():
            raise ValueError("Unpaired samples within geographic group")
        differences_by_group.append(np.mean([
            candidate_samples[sample] - baseline_samples[sample] for sample in candidate_samples
        ]))
    differences = np.asarray(differences_by_group)
    low, high = bootstrap_ci(differences)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "budget_index": budget_index,
        "n_groups": len(groups),
        "mean_difference": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
        "paired_t_p": float(stats.ttest_1samp(differences, 0).pvalue) if differences.std(ddof=1) else 1.0,
        "wilcoxon_p": float(stats.wilcoxon(differences, zero_method="zsplit").pvalue),
    }


def advancement_decision(rows: list[dict[str, str]]) -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    eligible: list[tuple[str, float]] = []
    for candidate in CANDIDATES:
        sus = [paired_group_comparison(rows, candidate, baseline, "sus") for baseline in CONTROLS]
        detector = paired_group_comparison(rows, candidate, "vqvae_random", "detector_retention")
        comparisons.extend((*sus, detector))
        if all(float(result["ci_low"]) > 0 for result in sus) and float(detector["ci_low"]) >= -0.02:
            eligible.append((candidate, min(float(result["ci_low"]) for result in sus)))
    selected = max(eligible, key=lambda entry: entry[1])[0] if eligible else None
    return {
        "decision": "advance_to_reserved_replication" if selected else "no_go",
        "selected_candidate": selected,
        "sampling_unit": "HLS geographic group",
        "primary_budget_index": 0,
        "comparisons": comparisons,
        "interpretation": "Development-screen result only; prior HLS selector work has used this partition.",
    }
