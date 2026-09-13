"""metrics.semantic_utility

Plain-English purpose: Research metrics such as Semantic Utility Score.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_ai.detector_base import MissionDetectorOutput


@dataclass(frozen=True)
class SemanticUtilityComponents:
    detector_retention: float
    object_retention: float
    relevance_retention: float
    region_preservation: float


class SemanticUtilityMetric:
    """Semantic Utility Score (SUS), returned on a 0-100 scale."""

    def __init__(
        self,
        detector_weight: float = 0.4,
        object_weight: float = 0.3,
        relevance_weight: float = 0.2,
        region_weight: float = 0.1,
    ) -> None:
        self.weights = np.asarray([detector_weight, object_weight, relevance_weight, region_weight], dtype="float64")
        self.weights = self.weights / max(float(self.weights.sum()), 1e-12)

    def score(
        self,
        detector_confidence: np.ndarray,
        relevance_map: np.ndarray,
        utility_map: np.ndarray,
        keep_mask: np.ndarray,
    ) -> tuple[float, SemanticUtilityComponents]:
        keep_mask = keep_mask.astype(bool)
        detector_retention = self._weighted_retention(detector_confidence, keep_mask)
        object_retention = self._object_retention(utility_map, keep_mask)
        relevance_retention = self._weighted_retention(relevance_map, keep_mask)
        region_preservation = self._region_preservation(utility_map, keep_mask)
        components = SemanticUtilityComponents(
            detector_retention=detector_retention,
            object_retention=object_retention,
            relevance_retention=relevance_retention,
            region_preservation=region_preservation,
        )
        values = np.asarray(
            [detector_retention, object_retention, relevance_retention, region_preservation],
            dtype="float64",
        )
        return float(np.clip(np.dot(self.weights, values) * 100.0, 0.0, 100.0)), components

    def score_before_after(
        self,
        before: MissionDetectorOutput,
        after: MissionDetectorOutput,
    ) -> tuple[float, SemanticUtilityComponents]:
        """Formal SUS from detector outputs before and after compression."""

        detector_retention = self._ratio(after.confidence_map.sum(), before.confidence_map.sum())
        object_retention = self._ratio(len(after.detections), len(before.detections))
        relevance_retention = self._ratio(after.utility_map.sum(), before.utility_map.sum())
        before_important = self._important_pixels(before.utility_map)
        after_important = self._important_pixels(after.utility_map, threshold=self._importance_threshold(before.utility_map))
        region_preservation = self._ratio(after_important, before_important)
        components = SemanticUtilityComponents(
            detector_retention=detector_retention,
            object_retention=object_retention,
            relevance_retention=relevance_retention,
            region_preservation=region_preservation,
        )
        values = np.asarray(
            [components.detector_retention, components.object_retention, components.relevance_retention, components.region_preservation],
            dtype="float64",
        )
        return float(np.clip(np.dot(self.weights, values) * 100.0, 0.0, 100.0)), components

    def _weighted_retention(self, values: np.ndarray, keep_mask: np.ndarray) -> float:
        total = float(values.sum())
        if total <= 1e-12:
            return float(keep_mask.mean())
        return float(np.clip(values[keep_mask].sum() / total, 0.0, 1.0))

    def _object_retention(self, utility_map: np.ndarray, keep_mask: np.ndarray) -> float:
        threshold = max(0.5, float(np.quantile(utility_map, 0.85))) if utility_map.size else 0.5
        object_mask = utility_map >= threshold
        total = int(object_mask.sum())
        if total == 0:
            return float(keep_mask.mean())
        return float(np.clip((object_mask & keep_mask).sum() / total, 0.0, 1.0))

    def _region_preservation(self, utility_map: np.ndarray, keep_mask: np.ndarray) -> float:
        high = utility_map >= float(utility_map.mean()) if utility_map.size else keep_mask
        if int(high.sum()) == 0:
            return float(keep_mask.mean())
        retained = (high & keep_mask).sum()
        return float(np.clip(retained / high.sum(), 0.0, 1.0))

    def _ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 1e-12:
            return 1.0 if numerator <= 1e-12 else 0.0
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    def _importance_threshold(self, utility_map: np.ndarray) -> float:
        if utility_map.size == 0:
            return 0.5
        return max(0.5, float(np.quantile(utility_map, 0.75)))

    def _important_pixels(self, utility_map: np.ndarray, threshold: float | None = None) -> int:
        if utility_map.size == 0:
            return 0
        threshold = self._importance_threshold(utility_map) if threshold is None else threshold
        return int((utility_map >= threshold).sum())
