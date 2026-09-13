# Space Semantic Compression Methodology

## Objective

This platform evaluates semantic token transmission for space and Earth observation imaging. The core hypothesis is that a spacecraft or edge node can preserve mission-relevant visual information while transmitting fewer bytes by ranking VQ-VAE tokens according to semantic importance.

## Implemented Pipeline

1. Image preprocessing resizes the frame to the VQ-VAE stride.
2. The VQ-VAE encoder converts the image into a discrete token grid.
3. The mission detector estimates Earth-observation utility for wildfire, flood, or ship-detection missions.
4. The semantic analyzer estimates token-level importance using saliency, region structure, and mission utility maps.
5. The utility-aware token selector optimizes `alpha*utility + beta*entropy - gamma*cost`.
6. A semantic keep-ratio simulates a constrained downlink budget.
6. Low-priority tokens are pruned and replaced by the modal token, while high-priority tokens are retained.
7. The decoder reconstructs the image from the semantic token set.
8. The satellite link simulator estimates packet counts, loss, latency, downlink time, and semantic fidelity.
9. The benchmark service exports CSV, JSON, hardware metadata, and optional TensorBoard logs.

## Current Baselines

- `jpeg_quality_45`: classical codec reference.
- `vqvae_full_tokens`: VQ-VAE reconstruction without semantic token pruning.
- `semantic_vqvae_45pct`: semantic token transmission with 45 percent token retention.
- `semantic_vqvae_25pct`: semantic token transmission with 25 percent token retention.
- `jpeg_quality_20` through `jpeg_quality_80`: decoded JPEG baselines with PSNR, SSIM, LPIPS, and SUS.
- `standard_autoencoder`, `variational_autoencoder`, and `lightweight_cnn`: evaluated automatically when trained checkpoints are present under `models/checkpoints/`.

Untrained baseline slots are reported as `not_implemented` instead of being assigned invalid metrics. This keeps the benchmark scientifically honest while documenting the planned comparison matrix.

## Metrics

- Compression ratio
- Bandwidth saved
- PSNR
- SSIM
- Semantic token count
- Semantic Utility Score
- Token entropy
- Energy per image
- Formal objective value
- Semantic fidelity retained
- Estimated downlink time
- Transmission time saved
- Inference latency by stage

## Reproducibility

Run a single-image benchmark:

```bash
python scripts/run_space_benchmark.py --image data/photos/val/img_000005.jpg
```

Results are written under:

```text
outputs/benchmarks/<benchmark_id>/
```

Each benchmark contains:

- `results.csv`
- `results.json`
- `benchmark.metadata.json`
- optional `tensorboard/`

## Edge Deployment Notes

The current inference path uses lazy model initialization and `torch.no_grad()`. The ONNX export script provides a starting point for Jetson and CubeSat-like edge runtime experiments:

```bash
python scripts/export_vqvae_onnx.py --checkpoint checkpoints/vqvae_s16k8.pt --output outputs/vqvae_reconstruct.onnx
```

Future work should validate ONNX numerical parity against the PyTorch checkpoint and add TensorRT profiling for Jetson-class hardware.
