"""token_selection.utility_pruner

Plain-English purpose: Algorithms that decide which learned image tokens are worth transmitting.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TokenSelectionWeights:
    alpha_utility: float = 0.65
    beta_entropy: float = 0.25
    gamma_cost: float = 0.10
    delta_detail: float = 0.0

    @classmethod
    def mission_utility(cls) -> "TokenSelectionWeights":
        return cls(alpha_utility=0.65, beta_entropy=0.25, gamma_cost=0.10, delta_detail=0.0)

    @classmethod
    def reconstruction_balanced(cls) -> "TokenSelectionWeights":
        return cls(alpha_utility=0.55, beta_entropy=0.20, gamma_cost=0.05, delta_detail=0.20)


class UtilityAwareTokenPruner:
    """Adaptive token selector using utility, entropy, image detail, and bandwidth cost."""

    def __init__(self, weights: TokenSelectionWeights | None = None) -> None:
        self.weights = weights or TokenSelectionWeights()

    def score_tokens(
        self,
        tokens: torch.Tensor,
        utility_map: np.ndarray,
        bandwidth_cost: np.ndarray | None = None,
        detail_map: np.ndarray | None = None,
    ) -> np.ndarray:
        utility = self._normalize(utility_map)
        token_shape = utility.shape
        entropy_proxy = self._local_token_entropy(tokens, token_shape)
        detail = self._normalize(detail_map) if detail_map is not None else self._edge_detail_map(utility)
        detail = self._match_shape(detail, token_shape)
        cost = bandwidth_cost if bandwidth_cost is not None else np.ones_like(utility, dtype="float32")
        cost = self._normalize(self._match_shape(cost, token_shape))
        score = (
            self.weights.alpha_utility * utility
            + self.weights.beta_entropy * entropy_proxy
            + self.weights.delta_detail * detail
            - self.weights.gamma_cost * cost
        )
        return self._normalize(score)

    def select(
        self,
        tokens: torch.Tensor,
        utility_map: np.ndarray,
        keep_ratio: float,
        bandwidth_cost: np.ndarray | None = None,
        detail_map: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_tokens(tokens, utility_map, bandwidth_cost=bandwidth_cost, detail_map=detail_map)
        total = scores.size
        keep_count = max(1, int(round(total * float(np.clip(keep_ratio, 0.01, 1.0)))))
        order = np.argsort(-scores.reshape(-1))
        mask = np.zeros(total, dtype=bool)
        mask[order[:keep_count]] = True
        return mask.reshape(scores.shape), scores

    def _edge_detail_map(self, values: np.ndarray) -> np.ndarray:
        image = values.astype("float32")
        if image.ndim != 2:
            image = image.reshape(values.shape[-2:])
        grad_y, grad_x = np.gradient(image)
        return self._normalize(np.sqrt(np.square(grad_x) + np.square(grad_y)))

    def _local_token_entropy(self, tokens: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
        values = tokens.detach().cpu().numpy().reshape(shape).astype("int64")
        entropy = np.zeros(shape, dtype="float32")
        padded = np.pad(values, 1, mode="edge")
        for row in range(shape[0]):
            for col in range(shape[1]):
                window = padded[row : row + 3, col : col + 3].reshape(-1)
                _, counts = np.unique(window, return_counts=True)
                probs = counts.astype("float64") / counts.sum()
                entropy[row, col] = float(-np.sum(probs * np.log2(probs + 1e-12)))
        return self._normalize(entropy)

    def _match_shape(self, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        values = values.astype("float32")
        if values.shape == shape:
            return values
        if values.ndim == 1 and values.size == shape[0] * shape[1]:
            return values.reshape(shape)
        row_idx = np.linspace(0, values.shape[0] - 1, shape[0]).round().astype("int64")
        col_idx = np.linspace(0, values.shape[1] - 1, shape[1]).round().astype("int64")
        return values[np.ix_(row_idx, col_idx)].astype("float32")

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = values.astype("float32")
        lo = float(values.min()) if values.size else 0.0
        hi = float(values.max()) if values.size else 0.0
        if hi - lo < 1e-8:
            return np.zeros_like(values, dtype="float32")
        return (values - lo) / (hi - lo)
