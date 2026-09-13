# Model Improvement Step 8: Large Sentinel-2 Training

## Purpose
This run trains the optional learned token-priority model on a larger real satellite dataset. The dataset contains 1,500 Sentinel-2/CEMS-derived patches generated from local Sentinel-2 wildfire imagery.

## Configuration
- Dataset directory: `datasets\sentinel2_1500_patch_256\images`
- Training images: 1500
- Epochs: 8
- Learning rate: 0.001
- Validation fraction: 0.2
- Train samples: 1,200
- Validation samples: 300
- Output checkpoint: `models\checkpoints\mode_conditioned_token_scorer_sentinel2_1500.pt`

## Result
- Final train distillation loss: 0.002844018095335438
- Final validation distillation loss: 0.003845291087636724

The validation loss remained close to the training loss, which suggests the learned selector is fitting the mode-conditioned teacher signal without obvious collapse on this split.

Training curve: `results/model_improvement_step8_large_satellite_training/training_loss_curve.png`

## Interpretation
This checkpoint is not automatically used by the API. It is a research artifact for evaluating whether a learned, mode-conditioned selector can outperform fixed hand-weighted token scoring.

The next required step is a held-out benchmark comparing this learned selector against the fixed mission-utility selector using SUS, detector retention, PSNR, SSIM, LPIPS, compression ratio, and bandwidth saved.
