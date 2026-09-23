"""Event-level inference for matched-rate wildfire codec experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ("sus", "detector_retention", "psnr", "ssim", "lpips")
BASELINES = ("jpeg", "jpeg2000", "vqvae_random", "vqvae_entropy")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def event_means(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, float]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        event = row.get("event_group", "").strip()
        if not event or event == "unknown":
            raise ValueError("Event-level analysis requires a verified event_group for every sample")
        budget = int(row["budget_index"])
        key = (row["method"], budget, event)
        sample_key = (row["sample_id"], budget, row["method"])
        if sample_key in seen:
            raise ValueError(f"Duplicate sample/method/budget: {sample_key}")
        seen.add(sample_key)
        grouped[key].append(row)
    output: dict[tuple[str, int, str], dict[str, float]] = {}
    for key, group in grouped.items():
        output[key] = {
            metric: float(np.mean([float(row[metric]) for row in group]))
            for metric in METRICS
            if all(row.get(metric) not in (None, "") and math.isfinite(float(row[metric])) for row in group)
        }
        output[key]["n_chips"] = float(len(group))
    return output


def bootstrap_ci(values: np.ndarray, seed: int = 20260922, repetitions: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    events = event_means(rows)
    budgets = sorted({key[1] for key in events})
    output: list[dict[str, float | int | str]] = []
    for budget in budgets:
        for metric in METRICS:
            family: list[dict[str, float | int | str]] = []
            for baseline in BASELINES:
                common = sorted(
                    event for method, index, event in events
                    if method == "vqvae_utility" and index == budget
                    and (baseline, budget, event) in events
                    and metric in events[(method, index, event)]
                    and metric in events[(baseline, budget, event)]
                )
                if len(common) < 2:
                    continue
                base = np.asarray([events[(baseline, budget, event)][metric] for event in common])
                candidate = np.asarray([events[("vqvae_utility", budget, event)][metric] for event in common])
                difference = candidate - base
                low, high = bootstrap_ci(difference, seed=20260922 + budget)
                t_result = stats.ttest_rel(candidate, base)
                w_result = stats.wilcoxon(difference, zero_method="zsplit")
                family.append({
                    "budget_index": budget,
                    "metric": metric,
                    "baseline": baseline,
                    "candidate": "vqvae_utility",
                    "n_events": len(common),
                    "baseline_mean": float(base.mean()),
                    "candidate_mean": float(candidate.mean()),
                    "mean_difference": float(difference.mean()),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "cohens_dz": float(difference.mean() / difference.std(ddof=1)) if difference.std(ddof=1) else 0.0,
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_p": float(w_result.pvalue),
                })
            for column in ("paired_t_p", "wilcoxon_p"):
                ordered = sorted(family, key=lambda row: float(row[column]))
                previous = 0.0
                for rank, result in enumerate(ordered):
                    adjusted = min(1.0, (len(ordered) - rank) * float(result[column]))
                    previous = max(previous, adjusted)
                    result[f"{column}_holm"] = previous
            output.extend(family)
    return output


def compare_checkpoints(
    baseline_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]
) -> list[dict[str, float | int | str]]:
    """Compare checkpoints only where the same image received the same byte budget."""
    baseline = {(row["sample_id"], row["method"], int(row["budget_index"])): row for row in baseline_rows}
    candidate = {(row["sample_id"], row["method"], int(row["budget_index"])): row for row in candidate_rows}
    if len(baseline) != len(baseline_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("Duplicate sample/method/budget rows")
    grouped: dict[tuple[str, int, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for key in sorted(baseline.keys() & candidate.keys()):
        old, new = baseline[key], candidate[key]
        if old["event_group"] != new["event_group"]:
            raise ValueError(f"Event mismatch for {key}")
        if int(old["target_bytes"]) != int(new["target_bytes"]):
            continue
        grouped[(key[1], key[2], old["event_group"])].append((old, new))
    output: list[dict[str, float | int | str]] = []
    for method, budget in sorted({(key[0], key[1]) for key in grouped}):
        groups = [pairs for (name, level, _), pairs in grouped.items() if name == method and level == budget]
        matched_chips = sum(len(pairs) for pairs in groups)
        baseline_chips = sum(key[1] == method and key[2] == budget for key in baseline)
        candidate_chips = sum(key[1] == method and key[2] == budget for key in candidate)
        if matched_chips != baseline_chips or matched_chips != candidate_chips:
            continue
        for metric in METRICS:
            differences = np.asarray([
                float(np.mean([float(new[metric]) - float(old[metric]) for old, new in pairs]))
                for pairs in groups
            ])
            if len(differences) < 2:
                continue
            low, high = bootstrap_ci(differences, seed=20260922 + budget)
            output.append({
                "method": method,
                "budget_index": budget,
                "metric": metric,
                "n_events": len(differences),
                "n_matched_chips": matched_chips,
                "mean_difference": float(differences.mean()),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "paired_t_p": float(stats.ttest_1samp(differences, 0).pvalue),
                "wilcoxon_p": float(stats.wilcoxon(differences, zero_method="zsplit").pvalue),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No paired event-level comparisons available")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
