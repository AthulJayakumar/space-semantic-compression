"""evaluation.statistics

Plain-English purpose: Benchmark pipelines, statistics, and publication-oriented experiments.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StatisticalTestResult:
    test: str
    statistic: float
    p_value: float
    effect_size: float


class StatisticalValidator:
    def paired_t_test(self, baseline: list[float], candidate: list[float]) -> StatisticalTestResult:
        baseline_arr, candidate_arr = self._paired_arrays(baseline, candidate)
        diff = candidate_arr - baseline_arr
        mean_diff = float(diff.mean())
        std_diff = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
        if diff.size < 2 or std_diff == 0:
            statistic, p_value = 0.0, 1.0
        else:
            try:
                from scipy import stats

                statistic, p_value = stats.ttest_rel(candidate_arr, baseline_arr)
            except Exception:
                statistic, p_value = mean_diff / (std_diff / np.sqrt(diff.size)), float("nan")
        effect = mean_diff / std_diff if std_diff else 0.0
        return StatisticalTestResult("paired_t_test", float(statistic), float(p_value), float(effect))

    def wilcoxon(self, baseline: list[float], candidate: list[float]) -> StatisticalTestResult:
        baseline_arr, candidate_arr = self._paired_arrays(baseline, candidate)
        diff = candidate_arr - baseline_arr
        try:
            from scipy import stats

            statistic, p_value = stats.wilcoxon(candidate_arr, baseline_arr, zero_method="zsplit")
        except Exception:
            statistic, p_value = float(np.abs(diff).sum()), float("nan")
        effect = float(np.median(diff) / (np.std(diff) + 1e-12))
        return StatisticalTestResult("wilcoxon_signed_rank", float(statistic), float(p_value), effect)

    def _paired_arrays(self, baseline: list[float], candidate: list[float]) -> tuple[np.ndarray, np.ndarray]:
        if len(baseline) != len(candidate):
            raise ValueError("Paired statistical tests require equal-length samples.")
        if len(baseline) == 0:
            raise ValueError("At least one paired sample is required.")
        return np.asarray(baseline, dtype="float64"), np.asarray(candidate, dtype="float64")
