"""Interleave high-scoring tokens across spatial blocks for coverage."""

from __future__ import annotations

import numpy as np


def stratified_order(scores: np.ndarray, block_size: int) -> np.ndarray:
    """Return a permutation that takes one token per block before a second."""
    if scores.ndim != 2 or block_size < 1:
        raise ValueError("Expected a 2D score map and positive block size")
    height, width = scores.shape
    blocks: list[np.ndarray] = []
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            positions = np.asarray([
                row * width + col
                for row in range(top, min(top + block_size, height))
                for col in range(left, min(left + block_size, width))
            ])
            ranked = positions[np.argsort(-scores.reshape(-1)[positions], kind="stable")]
            blocks.append(ranked)
    output: list[int] = []
    for layer in range(max(len(block) for block in blocks)):
        available = [block[layer] for block in blocks if layer < len(block)]
        available.sort(key=lambda position: (-float(scores.reshape(-1)[position]), int(position)))
        output.extend(int(position) for position in available)
    return np.asarray(output, dtype=np.int64)
