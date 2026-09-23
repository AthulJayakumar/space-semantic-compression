"""backend.services.token_transmission_service

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import torch


class TokenTransmissionService:
    def importance_to_token_map(self, importance_map: np.ndarray, token_shape: tuple[int, int]) -> np.ndarray:
        if importance_map.shape == token_shape:
            return importance_map.astype("float32")
        from PIL import Image

        image = Image.fromarray((importance_map * 255).astype("uint8"))
        resized = image.resize((token_shape[1], token_shape[0]))
        return np.asarray(resized).astype("float32") / 255.0

    def rank_tokens(self, tokens: torch.Tensor, token_importance: np.ndarray) -> np.ndarray:
        flat_importance = token_importance.reshape(-1)
        tie_breaker = np.arange(flat_importance.size) / max(flat_importance.size, 1) * 1e-6
        return np.argsort(-(flat_importance + tie_breaker))

    def build_keep_mask(self, token_importance: np.ndarray, keep_ratio: float) -> np.ndarray:
        keep_ratio = float(np.clip(keep_ratio, 0.01, 1.0))
        total = token_importance.size
        keep_count = max(1, int(round(total * keep_ratio)))
        order = self.rank_tokens(torch.empty(0), token_importance)
        mask = np.zeros(total, dtype=bool)
        mask[order[:keep_count]] = True
        return mask.reshape(token_importance.shape)

    def prune_tokens(self, tokens: torch.Tensor, keep_mask: np.ndarray) -> torch.Tensor:
        pruned = tokens.detach().clone()
        token_values = pruned.cpu().numpy().reshape(-1)
        values, counts = np.unique(token_values, return_counts=True)
        fallback_token = int(values[np.argmax(counts)])
        mask = torch.from_numpy(keep_mask).to(device=pruned.device, dtype=torch.bool)
        while mask.dim() < pruned.dim():
            mask = mask.unsqueeze(0)
        pruned = torch.where(mask, pruned, torch.full_like(pruned, fallback_token))
        return pruned.long()

    def token_entropy_bits(self, tokens: torch.Tensor) -> float:
        values = tokens.detach().cpu().numpy().reshape(-1)
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts.astype("float64") / max(values.size, 1)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return float(max(entropy, 0.0))

    def estimate_payload_kb(self, tokens: torch.Tensor, keep_mask: np.ndarray | None = None) -> float:
        return self.payload_bytes(tokens, keep_mask) / 1024.0

    def serialize_payload(self, tokens: torch.Tensor, keep_mask: np.ndarray | None = None) -> bytes:
        """Serialize exactly what would be transmitted for a token payload.

        Matched-rate experiments must compare encoded byte streams rather than
        theoretical token counts.  Keeping serialization in one place also
        ensures that API compression metrics and research benchmarks use the
        same wire representation.
        """
        codes = tokens.detach().cpu().numpy().astype(np.uint16)
        buffer = BytesIO()
        if keep_mask is None:
            np.savez_compressed(buffer, codes=codes)
        else:
            np.savez_compressed(buffer, codes=codes.reshape(-1)[keep_mask.reshape(-1)], mask=keep_mask.astype(np.uint8))
        return buffer.getvalue()

    def deserialize_payload(self, payload: bytes) -> tuple[torch.Tensor, np.ndarray | None]:
        """Recover tokens using only information actually present on the wire."""
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys == {"codes"}:
                codes = np.asarray(archive["codes"])
                if codes.ndim != 3 or not np.issubdtype(codes.dtype, np.integer):
                    raise ValueError("Full token payload must contain an integer [B,H,W] grid")
                return torch.from_numpy(codes.astype(np.int64)), None
            if keys != {"codes", "mask"}:
                raise ValueError("Token payload must contain codes and optional mask only")
            selected = np.asarray(archive["codes"])
            raw_mask = np.asarray(archive["mask"])
        if raw_mask.ndim != 2 or not np.isin(raw_mask, (0, 1)).all():
            raise ValueError("Token mask must be a binary [H,W] grid")
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("Selected token codes must be a flat integer array")
        mask = raw_mask.astype(bool)
        if selected.size != int(mask.sum()) or selected.size == 0:
            raise ValueError("Selected code count must equal the number of kept positions")
        values, counts = np.unique(selected, return_counts=True)
        fallback = int(values[np.argmax(counts)])
        restored = np.full(mask.shape, fallback, dtype=np.int64)
        restored[mask] = selected.astype(np.int64)
        return torch.from_numpy(restored).unsqueeze(0), mask

    def payload_bytes(self, tokens: torch.Tensor, keep_mask: np.ndarray | None = None) -> int:
        """Return the measured serialized payload size in bytes."""

        return len(self.serialize_payload(tokens, keep_mask))

    def semantic_fidelity_percent(self, token_importance: np.ndarray, keep_mask: np.ndarray) -> float:
        total = float(token_importance.sum())
        if total <= 0:
            return float(keep_mask.mean() * 100.0)
        retained = float(token_importance[keep_mask].sum())
        return float(np.clip(retained / total * 100.0, 0.0, 100.0))
