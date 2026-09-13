"""semantic_ai.detector_base

Plain-English purpose: Mission-specific detectors that turn images into utility maps.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area_percent: float


@dataclass(frozen=True)
class MissionDetectorOutput:
    mission: str
    utility_map: np.ndarray
    confidence_map: np.ndarray
    relevance_map: np.ndarray
    detections: list[DetectionResult] = field(default_factory=list)
    backend: str = "vision"


class MissionDetector:
    mission_name = "generic"

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self._yolo_model = None

    def detect(self, image: Image.Image) -> MissionDetectorOutput:
        yolo_output = self._try_yolo(image)
        if yolo_output is not None:
            return yolo_output
        return self._vision_detect(image.convert("RGB"))

    def _try_yolo(self, image: Image.Image) -> MissionDetectorOutput | None:
        if self.weights_path is None or not self.weights_path.exists():
            return None
        try:
            from ultralytics import YOLO
        except Exception:
            return None
        if self._yolo_model is None:
            self._yolo_model = YOLO(str(self.weights_path))
        result = self._yolo_model.predict(image, verbose=False)[0]
        width, height = image.size
        confidence_map = np.zeros((height, width), dtype="float32")
        relevance_map = np.zeros_like(confidence_map)
        detections: list[DetectionResult] = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            confidence = float(box.conf[0])
            label_idx = int(box.cls[0])
            label = result.names.get(label_idx, f"class_{label_idx}")
            confidence_map[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = np.maximum(
                confidence_map[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)],
                confidence,
            )
            relevance = self._label_relevance(label)
            relevance_map[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = np.maximum(
                relevance_map[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)],
                relevance,
            )
            area = max(0, x2 - x1) * max(0, y2 - y1)
            detections.append(
                DetectionResult(
                    label=label,
                    confidence=round(confidence, 4),
                    bbox=(x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
                    area_percent=round(area / max(width * height, 1) * 100.0, 4),
                )
            )
        utility_map = self._normalize(0.65 * confidence_map + 0.35 * relevance_map)
        return MissionDetectorOutput(
            mission=self.mission_name,
            utility_map=utility_map,
            confidence_map=confidence_map,
            relevance_map=relevance_map,
            detections=detections,
            backend="yolo",
        )

    def _vision_detect(self, image: Image.Image) -> MissionDetectorOutput:
        raise NotImplementedError

    def _label_relevance(self, label: str) -> float:
        return 1.0

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = values.astype("float32")
        lo = float(values.min()) if values.size else 0.0
        hi = float(values.max()) if values.size else 0.0
        if hi - lo < 1e-8:
            return np.zeros_like(values, dtype="float32")
        return (values - lo) / (hi - lo)

    def _connected_components(self, mask: np.ndarray, label: str, confidence_map: np.ndarray) -> list[DetectionResult]:
        try:
            import cv2
        except Exception:
            return []
        height, width = mask.shape
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
        detections: list[DetectionResult] = []
        image_area = max(height * width, 1)
        for idx in range(1, num):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < max(16, image_area * 0.0005):
                continue
            component_confidence = float(confidence_map[labels == idx].mean())
            detections.append(
                DetectionResult(
                    label=label,
                    confidence=round(min(0.99, component_confidence), 4),
                    bbox=(x, y, w, h),
                    area_percent=round(area / image_area * 100.0, 4),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)
