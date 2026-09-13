"""backend.services.encoder_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from backend.utils.tensor_utils import resolve_device
from src.models.vqvae import VQVAE

logger = logging.getLogger(__name__)


class EncoderService:
    """Lazy VQ-VAE loader and token encoder."""

    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device)
        self._model: VQVAE | None = None
        self._config: dict[str, Any] | None = None

    @property
    def model(self) -> VQVAE:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            _ = self.model
        return self._config or {"codebook_size": 8192, "code_dim": 256, "stride": 16}

    @property
    def stride(self) -> int:
        return int(self.config["stride"])

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode(image_tensor).long()

    def _load_model(self) -> VQVAE:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info("Loading VQ-VAE checkpoint from %s on %s", self.checkpoint_path, self.device)
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        args = checkpoint.get("args", {})
        self._config = {
            "codebook_size": args.get("codes", 8192),
            "code_dim": args.get("dim", 256),
            "stride": args.get("stride", 16),
        }
        model = VQVAE(**self._config).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model
