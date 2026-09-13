# Model Improvement Step 9: Full Sentinel-2 Training

## Purpose
This experiment trains the optional learned, mode-conditioned token-priority model on the full locally prepared Sentinel-2/CEMS patch set. The aim is to test whether more real satellite data improves the learned selector before using it in held-out compression benchmarks.

## Dataset
- Source imagery: local Sentinel-2/CEMS wildfire imagery
- Prepared patches: 3,224
- Patch size: 256 x 256
- Training samples: 2,579
- Validation samples: 645
- Validation split: 20%

## Training Configuration
- Model: `ModeConditionedTokenScorer`
- Training objective: teacher distillation from the validated fixed selectors
- Modes learned:
  - `mission_utility`
  - `reconstruction_balanced`
- Epochs: 12
- Learning rate: 0.001
- Device: CUDA GPU
- Output checkpoint: `models/checkpoints/mode_conditioned_token_scorer_sentinel2_full.pt`

## Results
| Run | Images | Train Samples | Validation Samples | Epochs | Final Train Loss | Final Validation Loss | Best Validation Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous 1,500-patch run | 1,500 | 1,200 | 300 | 8 | 0.002844 | 0.003845 | 0.003824 |
| Full 3,224-patch run | 3,224 | 2,579 | 645 | 12 | 0.002395 | 0.002883 | 0.002847 |

## Improvement Over Previous Training
- Final validation loss improved from 0.003845 to 0.002883.
- Best validation loss improved from 0.003824 to 0.002847.
- This is approximately a 25% validation-loss reduction.
- Final training loss improved from 0.002844 to 0.002395.

## Interpretation
The full-dataset training run improves the learned selector's ability to reproduce the validated token-priority teachers on held-out Sentinel-2 patches. This is a meaningful model-training improvement because the validation loss decreases while the validation set is larger than before.

However, this result only proves better selector distillation. It does not yet prove better compression performance. The next required experiment is a held-out compression benchmark comparing:

- fixed mission-utility selector
- learned selector trained on 1,500 patches
- learned selector trained on the full 3,224-patch dataset

The benchmark must report SUS, detector retention, PSNR, SSIM, LPIPS, compression ratio, and bandwidth saved before claiming end-to-end research improvement.

