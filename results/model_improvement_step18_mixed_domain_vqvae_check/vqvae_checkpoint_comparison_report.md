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
| cems_hls | cems_only | 80 | 97.321 | 0.9834 | 19.250 | 0.8476 | 547.00 | 99.82% |
| cems_hls | mixed_domain | 80 | 96.422 | 0.9623 | 18.887 | 0.8196 | 542.37 | 99.81% |
| cems_hls | original | 80 | 95.527 | 0.9863 | 18.782 | 0.8296 | 555.72 | 99.82% |
| sentinel2_patches | cems_only | 80 | 68.246 | 0.6367 | 23.333 | 0.8520 | 63.42 | 98.42% |
| sentinel2_patches | mixed_domain | 80 | 81.499 | 0.8038 | 25.255 | 0.8793 | 62.71 | 98.40% |
| sentinel2_patches | original | 80 | 78.852 | 0.8122 | 23.564 | 0.8612 | 63.95 | 98.43% |

## Paired Statistics
| Dataset | Candidate | Metric | Mean Difference | t-test p | 95% CI |
|---|---|---|---:|---:|---:|
| cems_hls | cems_only | semantic_utility_score | 1.793981 | 0.00553285 | [0.591106, 3.019559] |
| cems_hls | cems_only | detector_retention | -0.002912 | 0.378774 | [-0.009063, 0.003942] |
| cems_hls | cems_only | psnr | 0.468277 | 0.000435865 | [0.249293, 0.744639] |
| cems_hls | cems_only | ssim | 0.018097 | 0.0353067 | [0.004615, 0.036839] |
| cems_hls | mixed_domain | semantic_utility_score | 0.895524 | 0.274107 | [-0.674059, 2.469537] |
| cems_hls | mixed_domain | detector_retention | -0.023998 | 0.000629128 | [-0.037304, -0.011036] |
| cems_hls | mixed_domain | psnr | 0.104852 | 0.587545 | [-0.284305, 0.465718] |
| cems_hls | mixed_domain | ssim | -0.009926 | 0.479623 | [-0.037482, 0.015096] |
| sentinel2_patches | cems_only | semantic_utility_score | -10.605908 | 1.22898e-06 | [-14.709869, -6.754878] |
| sentinel2_patches | cems_only | detector_retention | -0.175439 | 2.54848e-11 | [-0.221356, -0.132042] |
| sentinel2_patches | cems_only | psnr | -0.231607 | 0.107048 | [-0.510926, 0.044915] |
| sentinel2_patches | cems_only | ssim | -0.009155 | 3.99784e-05 | [-0.013343, -0.005084] |
| sentinel2_patches | mixed_domain | semantic_utility_score | 2.647721 | 0.0156117 | [0.551462, 4.645050] |
| sentinel2_patches | mixed_domain | detector_retention | -0.008331 | 0.489525 | [-0.031932, 0.013997] |
| sentinel2_patches | mixed_domain | psnr | 1.690875 | 9.04355e-12 | [1.289951, 2.093884] |
| sentinel2_patches | mixed_domain | ssim | 0.018118 | 1.64873e-07 | [0.012254, 0.024459] |

## Interpretation
The fine-tuned checkpoint should only be promoted if it improves satellite reconstruction and utility without unacceptable detector-retention loss on other validation datasets.
