"""backend.services.profiling_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class InferenceProfiler:
    device: torch.device
    timings: dict[str, float] = field(default_factory=dict)
    token_count: int = 0

    @contextmanager
    def track(self, name: str):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timings[name] = (time.perf_counter() - start) * 1000.0

    def total_ms(self) -> float:
        return float(sum(self.timings.values()))

    def token_generation_rate(self) -> float:
        encoding_ms = self.timings.get("encoding", 0.0)
        if encoding_ms <= 0:
            return 0.0
        return self.token_count / (encoding_ms / 1000.0)
