"""backend.services.semantic_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from backend.schemas.response_schema import SemanticRegion

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency guard
    cv2 = None


@dataclass(frozen=True)
class SemanticAnalysis:
    importance_map: np.ndarray
    regions: list[SemanticRegion]
    method: str


class SemanticService:
    """Region-aware semantic analysis for space and edge imaging."""

    HIGH_PRIORITY_LABELS = {
        "fire": 1.0,
        "flood_water": 0.9,
        "urban_or_infrastructure": 0.86,
        "vessel_or_vehicle_candidate": 0.82,
        "vegetation_boundary": 0.65,
    }
    LOW_PRIORITY_LABELS = {
        "cloud": 0.25,
        "ocean_or_lake": 0.22,
        "sky_or_haze": 0.18,
        "empty_terrain": 0.35,
    }

    def analyze(self, image: Image.Image, token_shape: tuple[int, int], method: str = "hybrid") -> SemanticAnalysis:
        rgb = np.asarray(image.convert("RGB"))
        maps: list[np.ndarray] = []

        if method in {"hybrid", "saliency"}:
            maps.append(self._saliency_map(rgb))
        if method in {"hybrid", "segmentation"}:
            maps.append(self._segmentation_importance(rgb))
        if method in {"hybrid", "object"}:
            maps.append(self._object_candidate_map(rgb))

        if not maps:
            maps.append(self._saliency_map(rgb))

        importance = np.mean(np.stack(maps, axis=0), axis=0)
        importance = self._normalize(importance)
        token_importance = self.resize_importance(importance, token_shape)
        regions = self._regions_from_importance(rgb, importance)
        return SemanticAnalysis(importance_map=token_importance, regions=regions, method=method)

    def estimate_importance_mask(self, image: Image.Image, token_shape: tuple[int, int]) -> np.ndarray:
        importance = self.analyze(image, token_shape).importance_map
        threshold = float(np.quantile(importance, 0.70))
        return importance >= threshold

    def semantic_token_count(self, image: Image.Image, token_shape: tuple[int, int]) -> int:
        return int(self.estimate_importance_mask(image, token_shape).sum())

    def detail_map(self, image: Image.Image, token_shape: tuple[int, int]) -> np.ndarray:
        """Return token-scale structural detail so compression preserves important boundaries."""
        gray = np.asarray(image.convert("L")).astype("float32") / 255.0
        if cv2 is not None:
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            detail = cv2.magnitude(grad_x, grad_y)
        else:
            grad_y, grad_x = np.gradient(gray)
            detail = np.sqrt(np.square(grad_x) + np.square(grad_y))
        return self.resize_importance(self._normalize(detail), token_shape)

    def resize_importance(self, importance: np.ndarray, token_shape: tuple[int, int]) -> np.ndarray:
        height, width = token_shape
        if cv2 is not None:
            resized = cv2.resize(importance.astype("float32"), (width, height), interpolation=cv2.INTER_AREA)
        else:
            resized = np.asarray(Image.fromarray((importance * 255).astype("uint8")).resize((width, height))) / 255.0
        return self._normalize(resized).astype("float32")

    def _saliency_map(self, rgb: np.ndarray) -> np.ndarray:
        if cv2 is None:
            gray = rgb.mean(axis=2).astype("float32") / 255.0
            return self._normalize(np.abs(gray - gray.mean()))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        edges = cv2.Canny(gray, 80, 160).astype("float32")
        return self._normalize(0.7 * laplacian + 0.3 * edges)

    def _segmentation_importance(self, rgb: np.ndarray) -> np.ndarray:
        arr = rgb.astype("float32") / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        brightness = arr.mean(axis=2)
        saturation = arr.max(axis=2) - arr.min(axis=2)

        fire = (r > 0.55) & (g > 0.20) & (b < 0.25)
        water = (b > g * 1.05) & (b > r * 1.15) & (brightness < 0.65)
        cloud = (brightness > 0.72) & (saturation < 0.18)
        vegetation = (g > r * 1.08) & (g > b * 1.04)
        urban = (saturation < 0.24) & (brightness > 0.28) & (brightness < 0.75) & ~cloud

        importance = np.full(brightness.shape, self.LOW_PRIORITY_LABELS["empty_terrain"], dtype="float32")
        importance[water] = self.LOW_PRIORITY_LABELS["ocean_or_lake"]
        importance[cloud] = self.LOW_PRIORITY_LABELS["cloud"]
        importance[vegetation] = self.HIGH_PRIORITY_LABELS["vegetation_boundary"]
        importance[urban] = self.HIGH_PRIORITY_LABELS["urban_or_infrastructure"]
        importance[fire] = self.HIGH_PRIORITY_LABELS["fire"]
        return importance

    def _object_candidate_map(self, rgb: np.ndarray) -> np.ndarray:
        saliency = self._saliency_map(rgb)
        if cv2 is None:
            return saliency
        mask = (saliency > np.quantile(saliency, 0.88)).astype("uint8")
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = np.zeros_like(saliency, dtype="float32")
        image_area = saliency.shape[0] * saliency.shape[1]
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if 12 <= area <= image_area * 0.25:
                candidates[labels == idx] = self.HIGH_PRIORITY_LABELS["vessel_or_vehicle_candidate"]
        return np.maximum(candidates, saliency * 0.35)

    def _regions_from_importance(self, rgb: np.ndarray, importance: np.ndarray) -> list[SemanticRegion]:
        if cv2 is None:
            return []
        threshold = max(0.45, float(np.quantile(importance, 0.82)))
        mask = (importance >= threshold).astype("uint8")
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions: list[SemanticRegion] = []
        image_area = rgb.shape[0] * rgb.shape[1]
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            width = int(stats[idx, cv2.CC_STAT_WIDTH])
            height = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < max(8, image_area * 0.0005):
                continue
            crop = rgb[y : y + height, x : x + width]
            region_priority = float(importance[labels == idx].mean())
            regions.append(
                SemanticRegion(
                    label=self._classify_region(crop, region_priority),
                    bbox=[x, y, width, height],
                    confidence=round(min(0.99, 0.45 + region_priority * 0.55), 4),
                    priority=round(region_priority, 4),
                    area_percent=round(area / image_area * 100.0, 4),
                )
            )
        return sorted(regions, key=lambda item: item.priority, reverse=True)[:32]

    def _classify_region(self, crop: np.ndarray, priority: float) -> str:
        if crop.size == 0:
            return "semantic_candidate"
        arr = crop.astype("float32") / 255.0
        r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
        brightness = float(arr.mean())
        saturation = float(arr.max(axis=2).mean() - arr.min(axis=2).mean())
        if r > 0.55 and g > 0.20 and b < 0.30:
            return "fire"
        if b > g * 1.05 and b > r * 1.15 and brightness < 0.65:
            return "flood_water"
        if brightness > 0.72 and saturation < 0.18:
            return "cloud"
        if saturation < 0.25 and priority > 0.55:
            return "urban_or_infrastructure"
        return "vessel_or_vehicle_candidate" if priority > 0.65 else "semantic_candidate"

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = values.astype("float32")
        min_value = float(values.min()) if values.size else 0.0
        max_value = float(values.max()) if values.size else 0.0
        if max_value - min_value < 1e-8:
            return np.zeros_like(values, dtype="float32")
        return (values - min_value) / (max_value - min_value)
