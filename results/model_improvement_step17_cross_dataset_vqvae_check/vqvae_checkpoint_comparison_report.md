# Cross-Dataset VQ-VAE Checkpoint Comparison

## Purpose
Compare the original VQ-VAE checkpoint against the satellite fine-tuned checkpoint while keeping the mission-utility token selector fixed.

## Configuration
- Token retention: 0.8
- Per-dataset limit: 80
- LPIPS skipped: True

## Summary
| Dataset | Checkpoint | Images | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cems_hls | cems_hls_finetuned | 80 | 97.321 | 0.9834 | 19.250 | 0.8476 | 547.00 | 99.82% |
| cems_hls | original | 80 | 95.527 | 0.9863 | 18.782 | 0.8296 | 555.72 | 99.82% |
| sentinel2_patches | cems_hls_finetuned | 80 | 68.246 | 0.6367 | 23.333 | 0.8520 | 63.42 | 98.42% |
| sentinel2_patches | original | 80 | 78.852 | 0.8122 | 23.564 | 0.8612 | 63.95 | 98.43% |

## Paired Statistics
| Dataset | Candidate | Metric | Mean Difference | t-test p | 95% CI |
|---|---|---|---:|---:|---:|
| cems_hls | cems_hls_finetuned | semantic_utility_score | 1.793981 | 0.00553285 | [0.591106, 3.019559] |
| cems_hls | cems_hls_finetuned | detector_retention | -0.002912 | 0.378774 | [-0.009063, 0.003942] |
| cems_hls | cems_hls_finetuned | psnr | 0.468277 | 0.000435865 | [0.249293, 0.744639] |
| cems_hls | cems_hls_finetuned | ssim | 0.018097 | 0.0353067 | [0.004615, 0.036839] |
| sentinel2_patches | cems_hls_finetuned | semantic_utility_score | -10.605908 | 1.22898e-06 | [-14.709869, -6.754878] |
| sentinel2_patches | cems_hls_finetuned | detector_retention | -0.175439 | 2.54848e-11 | [-0.221356, -0.132042] |
| sentinel2_patches | cems_hls_finetuned | psnr | -0.231607 | 0.107048 | [-0.510926, 0.044915] |
| sentinel2_patches | cems_hls_finetuned | ssim | -0.009155 | 3.99784e-05 | [-0.013343, -0.005084] |

## Interpretation
The fine-tuned checkpoint should only be promoted if it improves satellite reconstruction and utility without unacceptable detector-retention loss on other validation datasets.
