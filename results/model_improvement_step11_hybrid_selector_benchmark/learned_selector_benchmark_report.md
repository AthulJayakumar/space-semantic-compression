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
| hybrid_fixed_0p50_learned_0p50 | 78.307 | 0.8325 | 21.542 | 0.8097 | 0.5837 | 70.49 | 98.57% |
| hybrid_fixed_0p60_learned_0p40 | 78.398 | 0.8336 | 21.584 | 0.8106 | 0.5836 | 70.51 | 98.57% |
| hybrid_fixed_0p70_learned_0p30 | 78.479 | 0.8342 | 21.607 | 0.8109 | 0.5836 | 70.52 | 98.57% |
| hybrid_fixed_0p80_learned_0p20 | 78.416 | 0.8319 | 21.616 | 0.8111 | 0.5835 | 70.52 | 98.57% |
| hybrid_fixed_0p90_learned_0p10 | 78.475 | 0.8320 | 21.621 | 0.8112 | 0.5834 | 70.54 | 98.57% |
| learned_1500_patch | 77.577 | 0.8248 | 21.154 | 0.8012 | 0.5846 | 70.47 | 98.57% |
| learned_full_3224_patch | 78.243 | 0.8294 | 21.450 | 0.8078 | 0.5845 | 70.42 | 98.57% |

## Paired Comparison Against Fixed Mission Utility
| Candidate | Metric | Mean Difference | Paired t-test p | Wilcoxon p | 95% CI |
|---|---|---:|---:|---:|---:|
| hybrid_fixed_0p50_learned_0p50 | semantic_utility_score | -0.216215 | 0.299117 | 0.219148 | [-0.627917, 0.170895] |
| hybrid_fixed_0p50_learned_0p50 | detector_retention | -0.001045 | 0.621477 | 0.314601 | [-0.004873, 0.003169] |
| hybrid_fixed_0p50_learned_0p50 | psnr | -0.096440 | 0.00441681 | 0.0176266 | [-0.170616, -0.036180] |
| hybrid_fixed_0p50_learned_0p50 | ssim | -0.001738 | 0.0158003 | 0.0025243 | [-0.003197, -0.000375] |
| hybrid_fixed_0p50_learned_0p50 | lpips | 0.000501 | 0.0728918 | 0.0503432 | [-0.000039, 0.001064] |
| hybrid_fixed_0p60_learned_0p40 | semantic_utility_score | -0.125174 | 0.52361 | 0.143612 | [-0.530254, 0.249438] |
| hybrid_fixed_0p60_learned_0p40 | detector_retention | 0.000004 | 0.998323 | 0.283957 | [-0.002977, 0.004168] |
| hybrid_fixed_0p60_learned_0p40 | psnr | -0.054745 | 0.0220768 | 0.0788111 | [-0.105892, -0.012504] |
| hybrid_fixed_0p60_learned_0p40 | ssim | -0.000915 | 0.0814113 | 0.00781551 | [-0.001943, 0.000052] |
| hybrid_fixed_0p60_learned_0p40 | lpips | 0.000378 | 0.0965296 | 0.0680368 | [-0.000037, 0.000830] |
| hybrid_fixed_0p70_learned_0p30 | semantic_utility_score | -0.043977 | 0.744785 | 0.176637 | [-0.285676, 0.235244] |
| hybrid_fixed_0p70_learned_0p30 | detector_retention | 0.000658 | 0.71558 | 0.356713 | [-0.002042, 0.004773] |
| hybrid_fixed_0p70_learned_0p30 | psnr | -0.032037 | 0.0856118 | 0.151947 | [-0.072839, 0.003390] |
| hybrid_fixed_0p70_learned_0p30 | ssim | -0.000545 | 0.209553 | 0.0158272 | [-0.001400, 0.000337] |
| hybrid_fixed_0p70_learned_0p30 | lpips | 0.000339 | 0.104116 | 0.039552 | [-0.000055, 0.000742] |
| hybrid_fixed_0p80_learned_0p20 | semantic_utility_score | -0.107485 | 0.286358 | 0.0246254 | [-0.306493, 0.081184] |
| hybrid_fixed_0p80_learned_0p20 | detector_retention | -0.001677 | 0.0140005 | 0.014934 | [-0.002992, -0.000463] |
| hybrid_fixed_0p80_learned_0p20 | psnr | -0.022640 | 0.131539 | 0.218158 | [-0.053804, 0.003162] |
| hybrid_fixed_0p80_learned_0p20 | ssim | -0.000339 | 0.328127 | 0.0219012 | [-0.000995, 0.000341] |
| hybrid_fixed_0p80_learned_0p20 | lpips | 0.000301 | 0.112955 | 0.143508 | [-0.000037, 0.000674] |
| hybrid_fixed_0p90_learned_0p10 | semantic_utility_score | -0.048225 | 0.502927 | 0.0635581 | [-0.183236, 0.093743] |
| hybrid_fixed_0p90_learned_0p10 | detector_retention | -0.001601 | 0.00412829 | 0.00126298 | [-0.002733, -0.000622] |
| hybrid_fixed_0p90_learned_0p10 | psnr | -0.017658 | 0.110252 | 0.62405 | [-0.041164, 0.001589] |
| hybrid_fixed_0p90_learned_0p10 | ssim | -0.000260 | 0.373323 | 0.0379384 | [-0.000856, 0.000299] |
| hybrid_fixed_0p90_learned_0p10 | lpips | 0.000200 | 0.112311 | 0.14232 | [-0.000030, 0.000458] |
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
