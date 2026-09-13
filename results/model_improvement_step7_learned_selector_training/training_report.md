# Model Improvement Step 7: Mode-Conditioned Token Scorer Training

## Purpose
This run trains the optional learned token-priority model by distilling the validated mission and reconstruction selectors.

## Configuration
- Dataset directory: `datasets/sentinel2_500_patch/images`
- Training images: 100
- Epochs: 10
- Learning rate: 0.001
- Output checkpoint: `models\checkpoints\mode_conditioned_token_scorer_sentinel2_100.pt`

## Result
- Final mean distillation loss: 0.004011540036299266

Training loss decreased from 0.023312 at epoch 1 to 0.004012 at epoch 10.

## Interpretation
This checkpoint is not automatically used by the API. It is a research artifact for evaluating whether a learned, mode-conditioned selector can outperform fixed hand-weighted token scoring.

The next required step is a learned-vs-fixed benchmark on held-out Sentinel-2 patches using SUS, detector retention, PSNR, SSIM, LPIPS, compression ratio, and bandwidth saved.
