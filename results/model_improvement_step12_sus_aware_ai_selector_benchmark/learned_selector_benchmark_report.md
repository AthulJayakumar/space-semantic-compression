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
| sus_aware_ai | 77.749 | 0.8166 | 21.107 | 0.7912 | 0.5923 | 71.26 | 98.59% |

## Paired Comparison Against Fixed Mission Utility
| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |
|---|---|---:|---:|---:|---:|
| sus_aware_ai | semantic_utility_score | -0.774123 | 0.363007 | 0.0248248 | [-2.357654, 1.065769] |
| sus_aware_ai | detector_retention | -0.016949 | 0.139911 | 0.00295494 | [-0.037954, 0.008581] |
| sus_aware_ai | psnr | -0.531952 | 0.00328578 | 0.0108293 | [-0.878416, -0.193650] |
| sus_aware_ai | ssim | -0.020322 | 8.38619e-06 | 1.72038e-07 | [-0.029040, -0.012264] |
| sus_aware_ai | lpips | 0.009048 | 3.26911e-11 | 3.91103e-11 | [0.006661, 0.011490] |

## Interpretation
- Best mean SUS: `fixed_mission_utility`.
- Best mean PSNR: `fixed_mission_utility`.
- A learned selector should only be promoted if it improves mission metrics such as SUS and detector retention, not only training loss.
