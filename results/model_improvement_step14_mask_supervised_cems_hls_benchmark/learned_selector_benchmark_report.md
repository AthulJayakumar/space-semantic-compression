# Learned Selector Held-Out Benchmark

## Purpose
This benchmark evaluates whether learned token-priority selectors improve end-to-end compression results on held-out Sentinel-2 patches.

## Configuration
- Dataset directory: `datasets\research_wildfire\cems_hls\cems_burnscars`
- Held-out images evaluated: 40
- Token retention ratio: 0.8
- Validation fraction: 0.2
- Seed: 1234
- LPIPS skipped: True

## Summary
| Selector | SUS | Detector Retention | PSNR | SSIM | LPIPS | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_mission_utility | 97.264 | 0.9991 | 17.917 | 0.8035 | n/a | 548.24 | 99.82% |
| mask_supervised_cems_hls | 95.489 | 0.9837 | 17.694 | 0.7884 | n/a | 541.01 | 99.81% |

## Paired Comparison Against Fixed Mission Utility
| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |
|---|---|---:|---:|---:|---:|
| mask_supervised_cems_hls | semantic_utility_score | -1.775342 | 0.00360619 | 0.00803844 | [-2.896988, -0.726782] |
| mask_supervised_cems_hls | detector_retention | -0.015362 | 0.0191016 | 0.0277078 | [-0.028323, -0.004387] |
| mask_supervised_cems_hls | psnr | -0.222301 | 0.259405 | 0.0793294 | [-0.597986, 0.145482] |
| mask_supervised_cems_hls | ssim | -0.015146 | 0.116298 | 0.000115358 | [-0.031359, 0.003492] |

## Interpretation
- Best mean SUS: `fixed_mission_utility`.
- Best mean PSNR: `fixed_mission_utility`.
- A learned selector should only be promoted if it improves mission metrics such as SUS and detector retention, not only training loss.
