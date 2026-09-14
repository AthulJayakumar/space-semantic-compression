# Cross-Dataset VQ-VAE Checkpoint Comparison

## Purpose
Compare the original VQ-VAE checkpoint against the satellite fine-tuned checkpoint while keeping the mission-utility token selector fixed.

## Configuration
- Token retention: 0.8
- Per-dataset limit: 500
- LPIPS skipped: True

## Summary
| Dataset | Checkpoint | Images | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sentinel2_patches | mixed_regularized | 500 | 81.159 | 0.8403 | 22.631 | 0.8354 | 68.60 | 98.53% |
| sentinel2_patches | original | 500 | 80.470 | 0.8642 | 21.305 | 0.8170 | 69.76 | 98.56% |

## Paired Statistics
| Dataset | Candidate | Metric | Mean Difference | t-test p | 95% CI |
|---|---|---|---:|---:|---:|
| sentinel2_patches | mixed_regularized | semantic_utility_score | 0.689497 | 0.174132 | [-0.234673, 1.661120] |
| sentinel2_patches | mixed_regularized | detector_retention | -0.023895 | 0.000104751 | [-0.034709, -0.012032] |
| sentinel2_patches | mixed_regularized | psnr | 1.326147 | 9.90787e-67 | [1.201452, 1.460468] |
| sentinel2_patches | mixed_regularized | ssim | 0.018399 | 1.14714e-18 | [0.014434, 0.022012] |

## Interpretation
The fine-tuned checkpoint should only be promoted if it improves satellite reconstruction and utility without unacceptable detector-retention loss on other validation datasets.
