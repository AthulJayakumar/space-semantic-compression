# SUS-Aware AI Token Selector Training

## Purpose
This run trains a learned token selector with a direct mission-utility surrogate rather than teacher imitation.

## Configuration
- Dataset directory: `datasets\sentinel2_full_patch_256\images`
- Training images: 3224
- Epochs: 8
- Learning rate: 0.0007
- Keep ratio: 0.8
- Budget weight: 3.0
- BCE weight: 0.2
- Validation fraction: 0.2
- Output checkpoint: `models\checkpoints\sus_aware_token_selector_sentinel2_full.pt`

## Result
- Final train loss: 0.35745690866998575
- Final validation loss: 0.3616629845874254
- Best validation loss: 0.3610826568317044
- Best validation soft SUS surrogate: 0.8819454227307046

## Interpretation
This checkpoint is experimental. It should only be promoted if held-out compression benchmarking improves SUS and detector retention against the fixed mission-utility selector.