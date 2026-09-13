# SUS-Aware AI Selector Experiment

## Purpose
This experiment tests whether a learned AI selector can improve over the fixed mission-utility selector by optimizing a direct Semantic Utility Score surrogate instead of imitating the fixed rule.

Two variants were tested:

1. `sus_aware_ai`: trained from random initialization using the SUS-aware surrogate.
2. `sus_aware_warm_ai`: initialized from the full 3,224-patch learned selector, then fine-tuned with the SUS-aware surrogate.

## Training Objective
The selector is trained to assign high keep probability to tokens with high:

- detector confidence
- wildfire utility
- relevance mass
- local image detail

The loss also enforces the target token-retention budget.

## Held-Out Benchmark Results
All results use 120 held-out Sentinel-2 patches at 80% token retention.

| Selector | SUS | Detector Retention | PSNR | SSIM | LPIPS | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed mission utility | 78.523 | 0.8336 | 21.639 | 0.8115 | 0.5832 | 70.54 | 98.57% |
| SUS-aware AI, scratch | 77.749 | 0.8166 | 21.107 | 0.7912 | 0.5923 | 71.26 | 98.59% |
| SUS-aware AI, warm-start | 77.593 | 0.8190 | 21.874 | 0.8055 | 0.5899 | 70.91 | 98.58% |

## Interpretation
The AI selector was successfully implemented and trained with a direct mission-utility objective. However, it does not yet outperform the fixed mission-utility selector on the primary mission metrics.

The scratch SUS-aware model improves compression ratio and bandwidth savings slightly, but loses detector retention, SUS, and reconstruction quality.

The warm-start SUS-aware model improves PSNR over the fixed selector, but still loses SUS, detector retention, SSIM, and LPIPS. This suggests the objective is currently pulling the model toward reconstruction/compression behavior rather than consistently preserving detector-visible wildfire utility.

## Decision
Do not replace the production/default selector.

The fixed mission-utility selector remains the best validated method for the current research pipeline. The SUS-aware AI selector should be presented as an experimental model-improvement path and as evidence that the platform supports trainable token-selection research.

## Next Research Direction
To make the AI selector competitive, the next step should be reinforcement-style or black-box optimization using actual post-reconstruction SUS, not only a differentiable pre-reconstruction surrogate.

