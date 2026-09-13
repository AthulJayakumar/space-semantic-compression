# Mode-Conditioned Token Scorer Training

## Purpose
This run trains the optional learned token-priority model by distilling the validated mission and reconstruction selectors.

## Configuration
- Dataset directory: `datasets\sentinel2_full_patch_256\images`
- Training images: 3224
- Epochs: 12
- Learning rate: 0.001
- Validation fraction: 0.2
- Output checkpoint: `models\checkpoints\mode_conditioned_token_scorer_sentinel2_full.pt`

## Result
- Final train distillation loss: 0.002395008138430079
- Final validation distillation loss: 0.002883081241565406

## Interpretation
This checkpoint is not automatically used by the API. It is a research artifact for evaluating whether a learned, mode-conditioned selector can outperform fixed hand-weighted token scoring.