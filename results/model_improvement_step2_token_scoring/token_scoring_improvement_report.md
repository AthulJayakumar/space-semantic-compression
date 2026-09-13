# Model Improvement Step 2: Detail-Aware Token Scoring

## Purpose
The previous utility-aware selector ranked tokens using semantic utility, token entropy, and transmission cost. This step adds a structural detail term so the selector gives extra priority to boundaries and high-frequency regions such as wildfire fronts, smoke edges, infrastructure outlines, and burn-scar contours.

## What Changed
- `TokenSelectionWeights` now includes `delta_detail`.
- `UtilityAwareTokenPruner.select(...)` accepts an optional `detail_map`.
- `SemanticService.detail_map(...)` computes a token-scale edge/detail map from the input image.
- `CompressionService.compress_image(...)` passes that detail map into token selection.
- Publication ablations now include `without_detail_term`.

## Controlled Token-Grid Result
| Method | Keep Ratio | Utility Mass Retained (%) | Boundary Tokens Retained | Boundary Retention (%) |
| --- | ---: | ---: | ---: | ---: |
| utility_entropy_only | 0.1 | 40.62 | 8 | 28.57 |
| detail_aware_utility | 0.1 | 40.62 | 26 | 92.86 |
| utility_entropy_only | 0.2 | 79.69 | 24 | 85.71 |
| detail_aware_utility | 0.2 | 79.69 | 28 | 100.0 |
| utility_entropy_only | 0.3 | 100.0 | 28 | 100.0 |
| detail_aware_utility | 0.3 | 100.0 | 28 | 100.0 |
| utility_entropy_only | 0.4 | 100.0 | 28 | 100.0 |
| detail_aware_utility | 0.4 | 100.0 | 28 | 100.0 |

## Interpretation
This controlled example isolates one model behavior: when many tokens have similar semantic utility, the detail-aware selector retains more boundary tokens. That is desirable for wildfire Earth Observation because mission-critical evidence is often located along edges, fronts, and contours rather than inside uniform regions.

This is not yet a dataset-level performance claim. The next validation step is rerunning the Sentinel-2 retention benchmark and comparing `full_system` against `without_detail_term` using SUS, detector retention, PSNR, SSIM, LPIPS, and bandwidth saved.

CSV output: `results\model_improvement_step2_token_scoring\token_scoring_detail_ablation.csv`
Figure output: `results\model_improvement_step2_token_scoring\detail_aware_token_selection.png`