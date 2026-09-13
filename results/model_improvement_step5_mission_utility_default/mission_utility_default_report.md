# Model Improvement Step 5: Mission-Utility Default Selector

## Purpose

The 500-patch Sentinel-2 ablation showed that detail-aware token scoring improves reconstruction metrics but slightly reduces wildfire mission utility. This step converts that evidence into the default system behavior.

## Decision

The default token selector is now **mission_utility**:

- utility weight: 0.65
- entropy weight: 0.25
- cost weight: 0.10
- detail weight: 0.00

The detail-aware selector remains available as **reconstruction_balanced**:

- utility weight: 0.55
- entropy weight: 0.20
- cost weight: 0.05
- detail weight: 0.20

## Why This Matters

The primary research hypothesis is about wildfire-relevant information preservation, not general visual reconstruction. Because the no-detail selector achieved slightly higher SUS and detector retention on the 500-patch Sentinel-2 ablation, it is the stronger default for the mission-utility objective.

## Implementation

- `TransmissionConfig.token_selection_mode` defaults to `mission_utility`.
- FastAPI accepts `token_selection_mode` for compression and transmission simulation.
- Streamlit exposes a mode selector.
- `custom` mode preserves ablation runners that inject experimental weights.
- Detail maps are computed only when the selected weights use a nonzero detail term.

## Research Interpretation

This is a useful negative/selection result. The detail term should not be presented as a universal improvement. It should be reported as a reconstruction-quality trade-off: better PSNR, SSIM, and LPIPS, but slightly weaker wildfire utility.
