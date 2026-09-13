"""backend.services.metrics_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from backend.utils.tensor_utils import image_to_metric_tensor


class MetricsService:
    def __init__(self) -> None:
        self._lpips_model = None

    def psnr(self, original: Image.Image, reconstructed: Image.Image) -> float:
        a, b = self._aligned_arrays(original, reconstructed)
        mse = float(np.mean((a - b) ** 2))
        if mse == 0:
            return float("inf")
        return 20.0 * math.log10(1.0 / math.sqrt(mse))

    def ssim(self, original: Image.Image, reconstructed: Image.Image) -> float:
        a, b = self._aligned_arrays(original, reconstructed, grayscale=True)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_a = float(a.mean())
        mu_b = float(b.mean())
        sigma_a = float(a.var())
        sigma_b = float(b.var())
        sigma_ab = float(((a - mu_a) * (b - mu_b)).mean())
        numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
        denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
        return float(numerator / denominator) if denominator else 0.0

    def lpips(self, original: Image.Image, reconstructed: Image.Image) -> float | None:
        try:
            import torch
            from lpips import LPIPS
        except Exception:
            return None
        if self._lpips_model is None:
            self._lpips_model = LPIPS(net="alex").eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_model = self._lpips_model.to(device)
        if original.size != reconstructed.size:
            original = original.resize(reconstructed.size, Image.Resampling.LANCZOS)
        with torch.no_grad():
            value = self._lpips_model(image_to_metric_tensor(original, device), image_to_metric_tensor(reconstructed, device))
        return float(value.mean().item())

    def _aligned_arrays(
        self,
        original: Image.Image,
        reconstructed: Image.Image,
        grayscale: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if original.size != reconstructed.size:
            original = original.resize(reconstructed.size, Image.Resampling.LANCZOS)
        if grayscale:
            original = original.convert("L")
            reconstructed = reconstructed.convert("L")
        return (
            np.asarray(original).astype("float32") / 255.0,
            np.asarray(reconstructed).astype("float32") / 255.0,
        )
