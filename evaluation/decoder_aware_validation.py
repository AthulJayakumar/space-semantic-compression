"""Development-only matched-rate comparison for decoder-aware token ranking."""

from __future__ import annotations

import numpy as np
import torch

from evaluation.matched_rate import MatchedRateBenchmark
from token_selection.decoder_aware import blended_decoder_aware_scores, decoder_impact_scores


class DecoderAwareMatchedRateBenchmark(MatchedRateBenchmark):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.methods = (*self.methods, "vqvae_decoder_impact", "vqvae_decoder_aware")

    def _rankings(
        self, tokens: torch.Tensor, utility_map: np.ndarray, detail_map: np.ndarray, sample_id: str
    ) -> dict[str, np.ndarray]:
        rankings = super()._rankings(tokens, utility_map, detail_map, sample_id)
        from token_selection.utility_pruner import UtilityAwareTokenPruner

        fixed = UtilityAwareTokenPruner(self.utility_weights).score_tokens(tokens, utility_map)
        embedding = self.service.encoder_service.model.vq.embedding.weight
        impact = decoder_impact_scores(tokens, embedding)
        blend = blended_decoder_aware_scores(fixed, impact)
        rankings["vqvae_decoder_impact"] = np.argsort(-impact.reshape(-1), kind="stable")
        rankings["vqvae_decoder_aware"] = np.argsort(-blend.reshape(-1), kind="stable")
        return rankings
