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
| sus_aware_warm_ai | 77.593 | 0.8190 | 21.874 | 0.8055 | 0.5899 | 70.91 | 98.58% |

## Paired Comparison Against Fixed Mission Utility
| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |
|---|---|---:|---:|---:|---:|
| sus_aware_warm_ai | semantic_utility_score | -0.929788 | 0.0757471 | 0.0373875 | [-1.968543, 0.107355] |
| sus_aware_warm_ai | detector_retention | -0.014622 | 0.0387996 | 0.0152706 | [-0.028519, -0.000992] |
| sus_aware_warm_ai | psnr | 0.235301 | 0.206102 | 0.281768 | [-0.130855, 0.586282] |
| sus_aware_warm_ai | ssim | -0.006022 | 0.12083 | 0.212551 | [-0.013532, 0.001348] |
| sus_aware_warm_ai | lpips | 0.006648 | 2.27156e-08 | 2.62007e-08 | [0.004552, 0.008819] |

## Interpretation
- Best mean SUS: `fixed_mission_utility`.
- Best mean PSNR: `sus_aware_warm_ai`.
- A learned selector should only be promoted if it improves mission metrics such as SUS and detector retention, not only training loss.
