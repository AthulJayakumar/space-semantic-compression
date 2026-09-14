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
| fixed_mission_utility | 97.166 | 0.9942 | 18.343 | 0.8164 | n/a | 554.45 | 99.82% |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | 97.082 | 0.9939 | 18.370 | 0.8164 | n/a | 554.56 | 99.82% |
| hybrid_fixed_0p60_mask_supervised_cems_hls_0p40 | 97.012 | 0.9943 | 18.364 | 0.8163 | n/a | 554.44 | 99.82% |
| hybrid_fixed_0p70_mask_supervised_cems_hls_0p30 | 97.046 | 0.9943 | 18.359 | 0.8164 | n/a | 554.46 | 99.82% |
| hybrid_fixed_0p80_mask_supervised_cems_hls_0p20 | 97.118 | 0.9943 | 18.347 | 0.8163 | n/a | 554.47 | 99.82% |
| hybrid_fixed_0p90_mask_supervised_cems_hls_0p10 | 97.197 | 0.9942 | 18.344 | 0.8163 | n/a | 554.46 | 99.82% |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | 97.227 | 0.9942 | 18.340 | 0.8163 | n/a | 554.37 | 99.82% |
| mask_supervised_cems_hls | 95.633 | 0.9801 | 18.108 | 0.8012 | n/a | 546.73 | 99.82% |

## Paired Comparison Against Fixed Mission Utility
| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |
|---|---|---:|---:|---:|---:|
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | semantic_utility_score | -0.084374 | 0.457281 | 0.398887 | [-0.309625, 0.137318] |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | detector_retention | -0.000273 | 0.567068 | 0.444587 | [-0.001232, 0.000651] |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | psnr | 0.027707 | 0.030951 | 0.00776919 | [0.001928, 0.051281] |
| hybrid_fixed_0p50_mask_supervised_cems_hls_0p50 | ssim | -0.000061 | 0.865893 | 0.806758 | [-0.000788, 0.000613] |
| hybrid_fixed_0p60_mask_supervised_cems_hls_0p40 | semantic_utility_score | -0.154599 | 0.159774 | 0.181202 | [-0.377172, 0.059372] |
| hybrid_fixed_0p60_mask_supervised_cems_hls_0p40 | detector_retention | 0.000118 | 0.69207 | 0.646462 | [-0.000481, 0.000725] |
| hybrid_fixed_0p60_mask_supervised_cems_hls_0p40 | psnr | 0.021639 | 0.0289409 | 0.00466043 | [0.001391, 0.039472] |
| hybrid_fixed_0p60_mask_supervised_cems_hls_0p40 | ssim | -0.000102 | 0.702962 | 0.784334 | [-0.000636, 0.000380] |
| hybrid_fixed_0p70_mask_supervised_cems_hls_0p30 | semantic_utility_score | -0.120398 | 0.182284 | 0.207757 | [-0.299555, 0.045684] |
| hybrid_fixed_0p70_mask_supervised_cems_hls_0p30 | detector_retention | 0.000190 | 0.341775 | 0.423596 | [-0.000167, 0.000591] |
| hybrid_fixed_0p70_mask_supervised_cems_hls_0p30 | psnr | 0.016760 | 0.0145652 | 0.00796196 | [0.003240, 0.029557] |
| hybrid_fixed_0p70_mask_supervised_cems_hls_0p30 | ssim | -0.000001 | 0.996189 | 0.52522 | [-0.000381, 0.000363] |
| hybrid_fixed_0p80_mask_supervised_cems_hls_0p20 | semantic_utility_score | -0.048385 | 0.530435 | 0.369497 | [-0.207356, 0.098155] |
| hybrid_fixed_0p80_mask_supervised_cems_hls_0p20 | detector_retention | 0.000135 | 0.41107 | 0.722108 | [-0.000144, 0.000485] |
| hybrid_fixed_0p80_mask_supervised_cems_hls_0p20 | psnr | 0.004231 | 0.398984 | 0.216299 | [-0.005915, 0.013126] |
| hybrid_fixed_0p80_mask_supervised_cems_hls_0p20 | ssim | -0.000163 | 0.210282 | 0.408852 | [-0.000428, 0.000080] |
| hybrid_fixed_0p90_mask_supervised_cems_hls_0p10 | semantic_utility_score | 0.030604 | 0.613558 | 0.761014 | [-0.087526, 0.152084] |
| hybrid_fixed_0p90_mask_supervised_cems_hls_0p10 | detector_retention | 0.000072 | 0.606927 | 0.858863 | [-0.000173, 0.000381] |
| hybrid_fixed_0p90_mask_supervised_cems_hls_0p10 | psnr | 0.001059 | 0.671507 | 0.75817 | [-0.003745, 0.005836] |
| hybrid_fixed_0p90_mask_supervised_cems_hls_0p10 | ssim | -0.000113 | 0.0581219 | 0.0995699 | [-0.000227, 0.000000] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | semantic_utility_score | 0.060614 | 0.290173 | 0.499656 | [-0.033480, 0.191296] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | detector_retention | 0.000077 | 0.494702 | 0.646462 | [-0.000068, 0.000323] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | psnr | -0.002145 | 0.161498 | 0.325952 | [-0.005299, 0.000726] |
| hybrid_fixed_0p95_mask_supervised_cems_hls_0p05 | ssim | -0.000107 | 0.0220778 | 0.048339 | [-0.000201, -0.000022] |
| mask_supervised_cems_hls | semantic_utility_score | -1.533032 | 0.000175786 | 0.00085869 | [-2.278373, -0.790297] |
| mask_supervised_cems_hls | detector_retention | -0.014022 | 0.000875879 | 0.000516715 | [-0.022346, -0.006737] |
| mask_supervised_cems_hls | psnr | -0.234865 | 0.105457 | 0.0124592 | [-0.514080, 0.028495] |
| mask_supervised_cems_hls | ssim | -0.015184 | 0.0211606 | 8.66723e-08 | [-0.026908, -0.002524] |

## Interpretation
- Best mean SUS: `hybrid_fixed_0p95_mask_supervised_cems_hls_0p05`.
- Best mean PSNR: `hybrid_fixed_0p50_mask_supervised_cems_hls_0p50`.
- A learned selector should only be promoted if it improves mission metrics such as SUS and detector retention, not only training loss.
