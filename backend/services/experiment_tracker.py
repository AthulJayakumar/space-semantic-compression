"""backend.services.experiment_tracker

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch


class ExperimentTracker:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def hardware_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            metadata["cuda_device"] = torch.cuda.get_device_name(0)
            metadata["cuda_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2)
        return metadata

    def write_metadata(self, experiment_dir: Path, name: str, payload: dict[str, Any]) -> Path:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        path = experiment_dir / f"{name}.metadata.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_tensorboard_scalars(self, experiment_dir: Path, scalars: dict[str, float]) -> str | None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            return None

        log_dir = experiment_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(log_dir))
        for key, value in scalars.items():
            writer.add_scalar(key, value, 0)
        writer.flush()
        writer.close()
        return str(log_dir)
