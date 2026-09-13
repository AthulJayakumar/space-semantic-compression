"""semantic_ai.wildfire_detector

Plain-English purpose: Mission-specific detectors that turn images into utility maps.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from semantic_ai.detector_base import MissionDetector, MissionDetectorOutput


class WildfireDetector(MissionDetector):
    mission_name = "wildfire_detection"

    def __init__(
        self,
        weights_path: str | Path | None = None,
        output_dir: str | Path | None = "figures/wildfire_utility",
        save_visualizations: bool = True,
    ) -> None:
        super().__init__(weights_path)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.save_visualizations = save_visualizations

    def detect(self, image: Image.Image) -> MissionDetectorOutput:
        output = super().detect(image)
        if self.save_visualizations and self.output_dir is not None:
            self.save_visualization(image, output)
        return output

    def _vision_detect(self, image: Image.Image) -> MissionDetectorOutput:
        arr = np.asarray(image.convert("RGB")).astype("float32") / 255.0
        return self._maps_from_rgb_array(arr)

    def detect_sentinel2(
        self,
        image: Image.Image,
        bands: dict[str, np.ndarray] | None = None,
    ) -> MissionDetectorOutput:
        arr = np.asarray(image.convert("RGB")).astype("float32") / 255.0
        output = self._maps_from_rgb_array(arr, bands=bands, backend="sentinel2_spectral_index")
        if self.save_visualizations and self.output_dir is not None:
            self.save_visualization(image, output)
        return output

    def _maps_from_rgb_array(
        self,
        arr: np.ndarray,
        bands: dict[str, np.ndarray] | None = None,
        backend: str = "vision_fire_index",
    ) -> MissionDetectorOutput:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        redness = np.clip(r - 0.55 * g - 0.35 * b, 0.0, 1.0)
        smoke = np.clip((arr.mean(axis=2) - arr.std(axis=2)) - 0.35, 0.0, 1.0)
        burn_scar = np.clip((r + g) * 0.35 - b * 0.55 - arr.mean(axis=2) * 0.15, 0.0, 1.0)
        thermal_proxy = np.zeros_like(redness)
        if bands:
            nir = self._band(bands, "B08", fallback=g)
            swir1 = self._band(bands, "B11", fallback=r)
            swir2 = self._band(bands, "B12", fallback=r)
            ndvi_loss = np.clip((g - nir) / (g + nir + 1e-6), 0.0, 1.0)
            nbr_loss = np.clip((swir2 - nir) / (swir2 + nir + 1e-6), 0.0, 1.0)
            burn_scar = self._normalize(0.45 * burn_scar + 0.35 * nbr_loss + 0.20 * ndvi_loss)
            thermal_proxy = self._normalize(0.55 * swir2 + 0.45 * swir1 - 0.35 * nir)
        confidence = self._normalize(0.56 * redness + 0.18 * smoke + 0.16 * burn_scar + 0.10 * thermal_proxy)
        mask = confidence > max(0.35, float(np.quantile(confidence, 0.92)))
        detections = self._connected_components(mask, "fire_or_smoke", confidence)
        relevance = self._normalize(confidence + 0.25 * smoke + 0.35 * burn_scar)
        utility = self._normalize(0.60 * confidence + 0.24 * relevance + 0.16 * burn_scar)
        return MissionDetectorOutput(
            mission=self.mission_name,
            utility_map=utility,
            confidence_map=confidence,
            relevance_map=relevance,
            detections=detections,
            backend=backend,
        )

    def _band(self, bands: dict[str, np.ndarray], key: str, fallback: np.ndarray) -> np.ndarray:
        value = bands.get(key)
        if value is None:
            return fallback
        arr = np.asarray(value).astype("float32")
        if arr.shape != fallback.shape:
            normalized = (self._normalize(arr) * 255).astype("uint8")
            arr = np.asarray(Image.fromarray(normalized).resize((fallback.shape[1], fallback.shape[0]), Image.Resampling.BILINEAR)).astype("float32")
        high = np.percentile(arr, 98)
        low = np.percentile(arr, 2)
        return np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)

    def _label_relevance(self, label: str) -> float:
        return 1.0 if any(key in label.lower() for key in ("fire", "smoke", "hotspot", "burn", "scar")) else 0.35

    def save_visualization(self, image: Image.Image, output: MissionDetectorOutput) -> dict[str, str]:
        if self.output_dir is None:
            return {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = uuid4().hex
        paths = {
            "utility_heatmap": self.output_dir / f"{stem}_utility.png",
            "confidence_map": self.output_dir / f"{stem}_confidence.png",
            "relevance_map": self.output_dir / f"{stem}_relevance.png",
            "overlay": self.output_dir / f"{stem}_overlay.png",
        }
        self._save_map(output.utility_map, paths["utility_heatmap"])
        self._save_map(output.confidence_map, paths["confidence_map"])
        self._save_map(output.relevance_map, paths["relevance_map"])
        heatmap = self._colorize(output.utility_map).resize(image.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(image.convert("RGB"), heatmap, 0.42)
        overlay.save(paths["overlay"])
        return {key: str(value) for key, value in paths.items()}

    def _save_map(self, values: np.ndarray, path: Path) -> None:
        self._colorize(values).save(path)

    def _colorize(self, values: np.ndarray) -> Image.Image:
        normalized = self._normalize(values)
        red = (normalized * 255).astype("uint8")
        green = ((1.0 - np.abs(normalized - 0.5) * 2.0) * 160).astype("uint8")
        blue = ((1.0 - normalized) * 255).astype("uint8")
        return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")
