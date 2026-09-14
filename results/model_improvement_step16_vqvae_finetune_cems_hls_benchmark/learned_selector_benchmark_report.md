# Learned Selector Held-Out Benchmark

## Purpose
This benchmark evaluates whether learned token-priority selectors improve end-to-end compression results on held-out Sentinel-2 patches.

## Configuration
- Dataset directory: `datasets\research_wildfire\cems_hls\cems_burnscars`
- Held-out images evaluated: 80
- Token retention ratio: 0.8
- Validation fraction: 0.2
- Seed: 1234
- LPIPS skipped: True

## Summary
| Selector | SUS | Detector Retention | PSNR | SSIM | LPIPS | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_mission_utility | 98.293 | 0.9824 | 18.736 | 0.8298 | n/a | 545.14 | 99.82% |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | 98.253 | 0.9819 | 18.775 | 0.8293 | n/a | 545.36 | 99.82% |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | 98.254 | 0.9823 | 18.739 | 0.8297 | n/a | 545.07 | 99.82% |
| mask_supervised_cems_hls | 96.437 | 0.9609 | 18.307 | 0.8027 | n/a | 538.41 | 99.81% |

## Paired Comparison Against Fixed Mission Utility
| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |
|---|---|---:|---:|---:|---:|
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | semantic_utility_score | -0.039989 | 0.657676 | 0.324612 | [-0.212880, 0.131124] |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | detector_retention | -0.000527 | 0.411286 | 0.301054 | [-0.001851, 0.000611] |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | psnr | 0.038890 | 0.00686154 | 0.00479912 | [0.012714, 0.065623] |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | ssim | -0.000412 | 0.227089 | 0.0783676 | [-0.001090, 0.000221] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | semantic_utility_score | -0.039096 | 0.118632 | 0.00321593 | [-0.090905, 0.009116] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | detector_retention | -0.000098 | 0.00833593 | 0.00314293 | [-0.000178, -0.000037] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | psnr | 0.002847 | 0.194579 | 0.469504 | [-0.001284, 0.007245] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | ssim | -0.000081 | 0.233148 | 0.263081 | [-0.000224, 0.000027] |
| mask_supervised_cems_hls | semantic_utility_score | -1.856267 | 0.000148273 | 0.00014075 | [-2.847393, -1.001068] |
| mask_supervised_cems_hls | detector_retention | -0.021510 | 0.000295917 | 9.50125e-05 | [-0.033149, -0.010962] |
| mask_supervised_cems_hls | psnr | -0.429222 | 0.00519659 | 0.000304309 | [-0.721027, -0.166113] |
| mask_supervised_cems_hls | ssim | -0.027072 | 5.4501e-05 | 5.16442e-11 | [-0.038474, -0.013586] |

## Interpretation
- Best mean SUS: `fixed_mission_utility`.
- Best mean PSNR: `hybrid_fixed_0p50_mask_supervised_cems_hls_0p50`.
- A learned selector should only be promoted if it improves mission metrics such as SUS and detector retention, not only training loss.
