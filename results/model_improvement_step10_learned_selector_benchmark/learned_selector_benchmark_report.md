# Learned Selector Held-Out Benchmark

## Purpose
This benchmark evaluates whether learned token-priority selectors improve end-to-end compression results on held-out Sentinel-2 patches.

## Configuration
- Dataset directory: `datasets\sentinel2_full_patch_256\images`
- Held-out images evaluated: 120
- Token retention ratio: 0.8
- Validation fraction: 0.2
- Seed: 1234
- LPIPS skipped: False

## Summary
| Selector | SUS | Detector Retention | PSNR | SSIM | LPIPS | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_mission_utility | 78.523 | 0.8336 | 21.639 | 0.8115 | 0.5832 | 70.54 | 98.57% |
| learned_1500_patch | 77.577 | 0.8248 | 21.154 | 0.8012 | 0.5846 | 70.47 | 98.57% |
| learned_full_3224_patch | 78.243 | 0.8294 | 21.450 | 0.8078 | 0.5845 | 70.42 | 98.57% |

## Paired Comparison Against Fixed Mission Utility
| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |
|---|---|---:|---:|---:|---:|
| learned_1500_patch | semantic_utility_score | -0.946675 | 0.00278501 | 0.0101245 | [-1.586008, -0.392651] |
| learned_1500_patch | detector_retention | -0.008793 | 0.00269402 | 0.00296976 | [-0.014471, -0.003321] |
| learned_1500_patch | psnr | -0.484820 | 5.0719e-10 | 2.18386e-11 | [-0.624516, -0.347463] |
| learned_1500_patch | ssim | -0.010249 | 1.15819e-08 | 1.85457e-11 | [-0.013489, -0.007093] |
| learned_1500_patch | lpips | 0.001364 | 0.00914744 | 0.0216491 | [0.000364, 0.002300] |
| learned_full_3224_patch | semantic_utility_score | -0.280459 | 0.366701 | 0.263442 | [-0.870526, 0.303584] |
| learned_full_3224_patch | detector_retention | -0.004196 | 0.240391 | 0.166809 | [-0.011504, 0.002791] |
| learned_full_3224_patch | psnr | -0.188614 | 0.000161376 | 0.00110736 | [-0.290243, -0.095746] |
| learned_full_3224_patch | ssim | -0.003713 | 0.000818249 | 0.00181756 | [-0.005884, -0.001658] |
| learned_full_3224_patch | lpips | 0.001246 | 0.00182555 | 0.0119399 | [0.000485, 0.002060] |

## Interpretation
- Best mean SUS: `fixed_mission_utility`.
- Best mean PSNR: `fixed_mission_utility`.
- A learned selector should only be promoted if it improves mission metrics such as SUS and detector retention, not only training loss.
