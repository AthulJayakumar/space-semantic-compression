# Model Improvement Step 4: 500-Patch Sentinel-2 Detail-Term Ablation

## Purpose
This experiment tests whether the detail-aware token selector improves real Sentinel-2 wildfire patch results compared with an ablated selector that removes the structural detail term.

## Experimental Setup
- Images evaluated: 500
- Token retention ratios: 0.80
- Full selector: utility + token entropy + detail + cost.
- Ablation: utility + token entropy + cost, with detail removed.
- Metrics: SUS, detector retention, compression ratio, bandwidth saved, PSNR, SSIM, and LPIPS.

## Summary
| Variant | Keep Ratio | Images | SUS | Detector Retention | Compression Ratio | Bandwidth Saved | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_detail_aware | 0.80 | 500 | 88.97 | 0.901 | 96.64x | 98.95% | 20.13 | 0.803 | 0.6007 |
| without_detail_term | 0.80 | 500 | 89.29 | 0.904 | 96.73x | 98.95% | 20.06 | 0.800 | 0.6020 |

## Paired Statistical Comparison
| Keep Ratio | Metric | Mean Difference | 95% Bootstrap CI | Paired t p | Wilcoxon p | Cohen's d |
|---:|---|---:|---:|---:|---:|---:|
| 0.80 | detector_retention | -0.0027 | [-0.0052, -0.0003] | 0.03252116 | 0.0286256 | -0.096 |
| 0.80 | lpips | -0.0013 | [-0.0016, -0.0011] | 0.0 | 0.0 | -0.507 |
| 0.80 | psnr | 0.0682 | [0.0492, 0.0888] | 0.0 | 0.0 | 0.309 |
| 0.80 | semantic_utility_score | -0.3176 | [-0.5190, -0.1061] | 0.00310191 | 0.00906137 | -0.133 |
| 0.80 | ssim | 0.0033 | [0.0026, 0.0040] | 0.0 | 0.0 | 0.420 |

## Interpretation
Positive differences mean the detail-aware selector improved the metric relative to the no-detail ablation. LPIPS should be interpreted separately because lower values are usually better.

This experiment is the first dataset-level check after the controlled token-grid result. If the gains are small or mixed, the scientific conclusion is still useful: detail-aware scoring improves boundary selection behavior, but its dataset-level value depends on detector sensitivity, token resolution, and reconstruction quality.

Figure: `results\model_improvement_step4_detail_ablation_500\detail_term_ablation_sus_detector.png`
