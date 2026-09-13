"""backend.services.visualization_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from backend.schemas.response_schema import SemanticRegion


class VisualizationService:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def save_semantic_heatmap(
        self,
        original: Image.Image,
        importance_map: np.ndarray,
        regions: list[SemanticRegion],
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"semantic_heatmap_{uuid4().hex}.png"
        heatmap = self._colorize_importance(importance_map).resize(original.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(original.convert("RGB"), heatmap, alpha=0.42)
        draw = ImageDraw.Draw(overlay)
        for region in regions[:16]:
            x, y, width, height = region.bbox
            color = (255, 80, 40) if region.priority >= 0.7 else (255, 220, 80)
            draw.rectangle([x, y, x + width, y + height], outline=color, width=2)
            draw.text((x, max(0, y - 12)), region.label, fill=color)
        overlay.save(target)
        return target

    def save_token_mask(self, keep_mask: np.ndarray) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"token_mask_{uuid4().hex}.png"
        mask = Image.fromarray((keep_mask.astype("uint8") * 255), mode="L").resize((512, 512), Image.Resampling.NEAREST)
        mask.convert("RGB").save(target)
        return target

    def _colorize_importance(self, importance_map: np.ndarray) -> Image.Image:
        normalized = np.clip(importance_map, 0.0, 1.0)
        red = (normalized * 255).astype("uint8")
        green = ((1.0 - np.abs(normalized - 0.5) * 2.0) * 180).astype("uint8")
        blue = ((1.0 - normalized) * 255).astype("uint8")
        return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")
