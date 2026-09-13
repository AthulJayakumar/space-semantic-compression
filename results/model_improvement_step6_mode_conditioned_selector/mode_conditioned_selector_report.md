# Model Improvement Step 6: Mode-Conditioned Learned Token Selector

## Purpose

The previous steps showed that fixed token-selection weights create a measurable trade-off between wildfire utility and reconstruction quality. This step starts improving the model itself by adding an optional learned token-priority network conditioned on the requested operating mode.

## New Research Component

`ModeConditionedTokenScorer` predicts one priority score per VQ-VAE token using:

- token ID embedding
- wildfire utility map
- local token entropy map
- structural detail map
- mode embedding

Supported modes:

- `mission_utility`
- `reconstruction_balanced`

## Why This Is Incremental

This does not replace the validated rule-based selector. The FastAPI backend and Streamlit dashboard still use the evidence-backed `mission_utility` default. The learned selector is introduced as a research component that can be trained, evaluated, and compared against the fixed selectors.

## Training Strategy

The first training runner uses teacher distillation:

1. Encode Sentinel-2 patches with the existing VQ-VAE.
2. Generate utility, entropy, and detail features.
3. Produce teacher token scores from the fixed mode selectors.
4. Train the learned scorer to predict mode-specific token-priority maps.

This is a conservative first step because it gives the model mode awareness without changing the VQ-VAE checkpoint or relying on unstable end-to-end training.

## Next Evaluation

After training, the learned selector should be evaluated against:

- fixed mission-utility selector
- fixed reconstruction-balanced selector
- random token selection
- entropy token selection

Metrics should include SUS, detector retention, PSNR, SSIM, LPIPS, compression ratio, and bandwidth saved.

## Current Status

Implemented and unit tested:

- optional PyTorch model
- mode embeddings
- token-grid scoring
- top-k token selection
- teacher-distillation training script
- smoke training run on 2 Sentinel-2 patches for 1 epoch, final mean loss 0.035119

Not yet completed:

- full Sentinel-2 training run
- 100/500-patch learned-selector benchmark
- integration as an API mode
