# Model Improvement Step 3: Sentinel-2 Detail-Term Ablation

## Purpose
This experiment tests whether the detail-aware token selector improves real Sentinel-2 wildfire patch results compared with an ablated selector that removes the structural detail term.

## Experimental Setup
- Images evaluated: 100
- Token retention ratios: 0.80
- Full selector: utility + token entropy + detail + cost.
- Ablation: utility + token entropy + cost, with detail removed.
- Metrics: SUS, detector retention, compression ratio, bandwidth saved, PSNR, SSIM, and LPIPS.

## Summary
| Variant | Keep Ratio | Images | SUS | Detector Retention | Compression Ratio | Bandwidth Saved | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_detail_aware | 0.80 | 100 | 85.53 | 0.878 | 93.80x | 98.93% | 21.81 | 0.855 | 0.5855 |
| without_detail_term | 0.80 | 100 | 85.47 | 0.877 | 93.82x | 98.93% | 21.74 | 0.854 | 0.5861 |

## Paired Statistical Comparison
| Keep Ratio | Metric | Mean Difference | 95% Bootstrap CI | Paired t p | Wilcoxon p | Cohen's d |
|---:|---|---:|---:|---:|---:|---:|
| 0.80 | detector_retention | 0.0017 | [-0.0024, 0.0059] | 0.45254014 | 0.29897873 | 0.075 |
| 0.80 | lpips | -0.0007 | [-0.0010, -0.0004] | 6.7e-06 | 8.15e-06 | -0.476 |
| 0.80 | psnr | 0.0644 | [0.0342, 0.0936] | 5.663e-05 | 2.1e-07 | 0.421 |
| 0.80 | semantic_utility_score | 0.0573 | [-0.2479, 0.3804] | 0.73539417 | 0.51727255 | 0.034 |
| 0.80 | ssim | 0.0017 | [0.0011, 0.0025] | 6.34e-06 | 1.4e-07 | 0.477 |

## Interpretation
Positive differences mean the detail-aware selector improved the metric relative to the no-detail ablation. LPIPS should be interpreted separately because lower values are usually better.

This experiment is the first dataset-level check after the controlled token-grid result. If the gains are small or mixed, the scientific conclusion is still useful: detail-aware scoring improves boundary selection behavior, but its dataset-level value depends on detector sensitivity, token resolution, and reconstruction quality.

Figure: `results\model_improvement_step3_detail_ablation\detail_term_ablation_sus_detector.png`