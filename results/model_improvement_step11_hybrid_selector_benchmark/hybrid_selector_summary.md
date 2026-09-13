# Model Improvement Step 11: Hybrid Selector Benchmark

## Question
Can a hybrid selector improve results by combining the validated fixed mission-utility selector with the learned full-dataset selector?

## Method
The benchmark combines normalized token-priority scores:

`hybrid_score = (1 - learned_weight) * fixed_score + learned_weight * learned_score`

The experiment tested learned-score weights of 0.10, 0.20, 0.30, 0.40, and 0.50 on the same 120 held-out Sentinel-2 validation patches used in the learned-selector benchmark.

## Main Result
| Selector | SUS | Detector Retention | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Fixed mission utility | 78.523 | 0.8336 | 21.639 | 0.8115 | 0.5832 |
| Best hybrid by SUS: 70% fixed + 30% learned | 78.479 | 0.8342 | 21.607 | 0.8109 | 0.5836 |
| Pure learned full selector | 78.243 | 0.8294 | 21.450 | 0.8078 | 0.5845 |

## Interpretation
The hybrid selector improves over the pure learned selector and almost matches the fixed selector. The 70% fixed / 30% learned hybrid gives the highest detector-retention mean among the hybrid settings, but the gain over the fixed selector is extremely small and not statistically significant.

The fixed mission-utility selector remains the best default because it still has the highest SUS, PSNR, SSIM, and LPIPS performance on this held-out test. The hybrid selector is promising as a research ablation, but it should not replace the fixed default unless future training or calibration produces a clear statistically significant gain.

## Recommendation
Keep `mission_utility` as the production and paper-default selector. Report the hybrid selector as an exploratory model-improvement experiment showing that learned scoring can approach the validated rule-based method, but not yet surpass it.

