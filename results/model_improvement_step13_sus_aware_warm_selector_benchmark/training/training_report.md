# SUS-Aware AI Token Selector Training

## Purpose
This run trains a learned token selector with a direct mission-utility surrogate rather than teacher imitation.

## Configuration
- Dataset directory: `datasets\sentinel2_full_patch_256\images`
- Training images: 3224
- Epochs: 5
- Learning rate: 0.0002
- Keep ratio: 0.8
- Budget weight: 3.0
- BCE weight: 0.2
- Initial checkpoint: `models\checkpoints\mode_conditioned_token_scorer_sentinel2_full.pt`
- Validation fraction: 0.2
- Output checkpoint: `models\checkpoints\sus_aware_token_selector_warm_sentinel2_full.pt`

## Result
- Final train loss: 0.358404846305115
- Final validation loss: 0.3605031302501989
- Best validation loss: 0.3605031302501989
- Best validation soft SUS surrogate: 0.8772290561088296

## Interpretation
This checkpoint is experimental. It should only be promoted if held-out compression benchmarking improves SUS and detector retention against the fixed mission-utility selector.