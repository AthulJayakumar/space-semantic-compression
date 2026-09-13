"""backend.services.decoder_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import torch

from backend.services.encoder_service import EncoderService


class DecoderService:
    def __init__(self, encoder_service: EncoderService) -> None:
        self.encoder_service = encoder_service

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder_service.model.decode(tokens.to(self.encoder_service.device))
