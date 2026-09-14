# Model Improvement Step 15: Mask-Supervised Hybrid Selector

## Purpose

Step 14 showed that a pure mask-supervised selector trained on CEMS-HLS learns
from burn-scar masks, but does not yet outperform the fixed mission-utility
selector on end-to-end wildfire semantic metrics. Step 15 tests whether the
learned selector can still add value as a small component inside a hybrid score.

The hybrid score is:

```text
hybrid_score = fixed_weight * fixed_mission_score + learned_weight * mask_supervised_score
```

## Dataset

- Dataset: CEMS-HLS burn-scar satellite imagery
- Held-out scenes: 80
- Token retention: 80%
- LPIPS: skipped for speed
- Compared selectors:
  - fixed mission utility
  - pure mask-supervised AI selector
  - hybrid weights from 5% to 50% learned contribution

## Key Result

| Selector | SUS | Detector Retention | PSNR | SSIM | Compression Ratio | Bandwidth Saved |
|---|---:|---:|---:|---:|---:|---:|
| Fixed mission utility | 97.166 | 0.9942 | 18.343 | 0.8164 | 554.45x | 99.82% |
| 95% fixed + 5% AI | **97.227** | 0.9942 | 18.340 | 0.8163 | 554.37x | 99.82% |
| 50% fixed + 50% AI | 97.082 | 0.9939 | **18.370** | 0.8164 | 554.56x | 99.82% |
| Pure mask-supervised AI | 95.633 | 0.9801 | 18.108 | 0.8012 | 546.73x | 99.82% |

## Statistical Interpretation

- The best mean SUS came from the 95% fixed + 5% AI hybrid, but the paired
  improvement over the fixed selector was not statistically significant:
  +0.0606 SUS, p=0.290, 95% CI [-0.0335, 0.1913].
- The 50% fixed + 50% AI hybrid produced a small statistically significant PSNR
  improvement:
  +0.0277 dB, p=0.031, 95% CI [0.0019, 0.0513].
- The pure mask-supervised AI selector significantly reduced SUS and detector
  retention, so it should not replace the fixed selector.

## Decision

The default production/research selector should remain `mission_utility`.

The strongest next candidate is a conservative hybrid:

```text
95% fixed mission utility + 5% mask-supervised AI
```

This hybrid should be treated as experimental because the SUS gain is small and
not yet statistically significant. The evidence suggests the learned model may
contain some useful signal, but the hand-weighted mission selector remains the
most reliable method for wildfire semantic preservation.

## Next Research Step

Train a second mask-supervised selector with the fixed mission-utility score as
a regularization target. The objective should reward agreement with the proven
mission selector while using masks only to refine ambiguous token rankings. This
is more promising than training the AI selector from masks alone.
