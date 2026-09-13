# Research Report: AI Semantic Compression for Space Imaging

## Executive Summary

This pilot experiment evaluates the current CompressAI space semantic compression platform on one validation image: `data/photos/val/img_000005.jpg`. The system uses a VQ-VAE token encoder, semantic region analysis, importance-aware token pruning, and a simulated satellite downlink model.

The strongest result is that semantic token transmission at 45 percent token retention improved compression ratio from `7.46x` to `10.75x`, increased bandwidth savings from `86.59%` to `90.70%`, and retained `82.54%` semantic fidelity. This comes with a reconstruction-quality tradeoff: PSNR decreased from `27.03 dB` to `18.56 dB`, while SSIM remained usable at `0.826`.

## Experimental Setup

- Image: `img_000005.jpg`
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU
- GPU memory: `6140.5 MB`
- Python: `3.11.14`
- PyTorch: `2.5.1+cu121`
- Platform: Windows 10
- Model: existing VQ-VAE checkpoint `checkpoints/vqvae_s16k8.pt`
- Token grid size: `16,384` tokens
- Satellite simulation link:
  - Bandwidth: `256 kbps`
  - Latency: `600 ms`
  - Packet loss: `2%`
  - Outage probability: `5%`
  - Packet size: `1024 bytes`

## Benchmark Results

| Method | Status | Compression Ratio | Bandwidth Saved | PSNR | SSIM | Semantic Fidelity | Token Entropy | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JPEG quality 45 | ok | `1.95x` | `48.74%` | n/a | n/a | n/a | n/a | n/a |
| VQ-VAE full tokens | ok | `7.46x` | `86.59%` | `27.03` | `0.9837` | `100.00%` | `5.5921 bits` | `1625.26 ms` |
| Semantic VQ-VAE 45% | ok | `10.75x` | `90.70%` | `18.56` | `0.8260` | `82.54%` | `3.6621 bits` | `989.64 ms` |
| Semantic VQ-VAE 25% | ok | `16.96x` | `94.10%` | `14.71` | `0.5835` | `59.98%` | `2.1873 bits` | `985.80 ms` |

## Detailed Semantic Transmission Result

For the 45 percent semantic-token setting:

- Original image size: `100.1055 KB`
- Semantic compressed payload: `9.3145 KB`
- Compression ratio: `10.7473x`
- Bandwidth saved: `90.70%`
- Semantic tokens transmitted: `7,373 / 16,384`
- Semantic regions detected: `19`
- PSNR: `18.5563 dB`
- SSIM: `0.8260`
- Token entropy: `3.6621 bits`
- Semantic fidelity retained: `82.54%`

### Satellite Downlink Simulation

- Full token payload: `13.2051 KB`
- Semantic token payload: `9.3145 KB`
- Packets required: `10`
- Estimated delivered packets: `9`
- Full-token downlink time: `1.0127 s`
- Semantic-token downlink time: `0.8911 s`
- Transmission time saved: `0.1216 s`

## Inference Profiling

| Stage | Latency |
|---|---:|
| Preprocessing | `87.65 ms` |
| Token encoding | `251.10 ms` |
| Semantic analysis | `403.64 ms` |
| Transmission simulation | `3.02 ms` |
| Reconstruction | `341.05 ms` |
| Postprocessing and visualization | `183.17 ms` |
| Total | `1269.64 ms` |

Token generation speed was `65,247.74 tokens/sec` on CUDA.

## Interpretation

The experiment supports the core research claim: semantic token pruning can substantially reduce transmitted payload while retaining a majority of mission-relevant information. The 45 percent semantic setting is the best current operating point because it improves compression and bandwidth use while preserving much higher structural quality than the 25 percent setting.

The 25 percent setting is useful as a stress test for extremely constrained links, but it drops semantic fidelity to `59.98%` and SSIM to `0.5835`, making it less suitable for high-confidence Earth observation analysis.

## Limitations

- This is a single-image pilot result, not a dataset-level benchmark.
- LPIPS is currently reported as `null` in the API path and should be enabled for perceptual evaluation.
- JPEG baseline does not yet compute PSNR/SSIM against its decoded reconstruction.
- Autoencoder, VAE, and lightweight CNN baselines are reserved but not yet trained, so they are correctly marked `not_implemented`.
- Semantic detection currently uses lightweight saliency/segmentation heuristics, not a trained satellite object detector.

## Research Conclusion

The current platform is a valid early research prototype for AI-powered semantic communication in space imaging. It demonstrates token-based semantic transmission, importance-aware compression, satellite-link simulation, and reproducible metric export. The most promising configuration in this pilot is `semantic_vqvae_45pct`, achieving `10.75x` compression and `90.70%` bandwidth savings while retaining `82.54%` semantic fidelity.

## Recommended Next Experiments

1. Run the benchmark across the full validation set.
2. Add decoded JPEG PSNR/SSIM for a fair classical-codec comparison.
3. Enable LPIPS in the inference API.
4. Add YOLO or satellite-specific object detection for buildings, ships, roads, floods, smoke, and wildfire.
5. Produce quality-vs-bandwidth curves for keep ratios from `10%` to `100%`.
6. Add Jetson-class profiling for edge deployment claims.

