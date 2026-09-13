# Space Semantic Compression

**Semantic Communication and Edge Intelligence for Resource-Constrained Earth Observation Satellites**

This repository contains a research prototype for **mission-aware satellite image compression**.  
In simple terms: it studies how a satellite can send the **most important parts of an image first** when bandwidth is limited.

The current mission focus is **wildfire monitoring**. Instead of compressing every pixel equally, the system tries to preserve regions that may contain:

- active fire
- smoke
- burn scars
- damaged or affected terrain
- mission-relevant Earth Observation evidence

The project is designed for PhD applications, supervisor review, research demonstrations, and future publication work in Earth Observation AI, semantic communication, edge AI, and satellite systems.

---

## 1. What Problem Does This Solve?

Satellites can capture huge amounts of imagery, but they cannot always send all of it back to Earth quickly. This is especially true for:

- CubeSats
- small satellites
- disaster-response missions
- low-bandwidth ground-station windows
- onboard AI systems with limited power and memory

For wildfire monitoring, a perfect-looking full image is not always the most urgent need. A user may need to know:

> Where is the fire? Where is the smoke? Which regions matter most?

This repository explores a system that compresses images according to **semantic utility**, meaning mission usefulness.

---

## 2. One-Sentence Summary

> This project transmits learned image tokens according to wildfire importance, preserving mission-critical Earth Observation information under severe bandwidth constraints.

---

## 3. Key Results So Far

The current benchmark uses DFire, FLAME, and Sentinel-2/CEMS wildfire-related imagery.

| Dataset | Images | SUS | Detector Retention | Bandwidth Saved | Compression Ratio |
|---|---:|---:|---:|---:|---:|
| DFire | 50 | 83.14 | 0.849 | 95.10% | 25.30x |
| FLAME | 18 | 65.84 | 0.834 | 96.27% | 55.51x |
| Sentinel-2/CEMS | 100 | 79.52 | 0.741 | 99.35% | 155.77x |

**SUS** means **Semantic Utility Score**.  
It measures how much wildfire-relevant information survives compression and reconstruction.

Additional 500-patch Sentinel-2 conventional codec validation:

| Baseline | Patches | SUS | Detector Retention | PSNR | SSIM | Bandwidth Saved | Compression Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| JPEG2000 rate-20 | 500 | 95.15 | 0.964 | 30.94 | 0.978 | 95.01% | 20.04x |
| CCSDS-style wavelet proxy Q24 | 500 | 82.74 | 0.881 | 25.35 | 0.934 | 93.34% | 23.93x |

The CCSDS-style result is a transform-coding proxy, **not** a certified CCSDS implementation.

Model improvement Step 1 identified a better operating point for the current utility-aware model:

| Utility-Aware Retention | SUS | Detector Retention | Bandwidth Saved | Compression Ratio |
|---:|---:|---:|---:|---:|
| 50% | 82.54 | 0.783 | 99.31% | 145.36x |
| **80% recommended** | **90.49** | **0.907** | **99.07%** | **107.77x** |
| 100% full VQ-VAE | 92.81 | 0.947 | 98.98% | 99.08x |

This improves the research story: 80% retention preserves most of the semantic utility of full VQ-VAE while keeping stronger compression than the full-token setting.

Model improvement Step 2 adds detail-aware token scoring. This gives extra priority to structural boundaries such as fire fronts, smoke edges, infrastructure outlines, and burn-scar contours when utility scores are tied.

| Controlled Token Test | Boundary Retention at 10% Tokens |
|---|---:|
| Utility + entropy only | 28.57% |
| Detail-aware utility scoring | 92.86% |

This is controlled component evidence, not yet a full dataset-level claim. The next benchmark step is to compare `full_system` against `without_detail_term` across Sentinel-2 retention experiments.

Model improvement Step 3 ran that dataset-level check on 100 Sentinel-2 patches at the recommended 80% token-retention operating point:

| 80% Retention Variant | SUS | Detector Retention | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Without detail term | 85.47 | 0.877 | 21.74 | 0.854 | 0.5861 |
| Detail-aware scoring | 85.53 | 0.878 | 21.81 | 0.855 | 0.5855 |

Interpretation: the detail term gives a small but statistically significant improvement in reconstruction quality (PSNR, SSIM, LPIPS) while leaving SUS and detector retention broadly unchanged. This is useful, bounded evidence: detail-aware scoring helps visual/structural preservation, but does not yet create a large wildfire-utility jump.

Model improvement Step 4 scaled the same ablation to **500 Sentinel-2 patches**:

| 80% Retention Variant | Images | SUS | Detector Retention | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Without detail term | 500 | 89.29 | 0.904 | 20.06 | 0.800 | 0.6020 |
| Detail-aware scoring | 500 | 88.97 | 0.901 | 20.13 | 0.803 | 0.6007 |

Interpretation: the 500-patch result confirms the trade-off. Detail-aware scoring improves PSNR, SSIM, and LPIPS, but slightly reduces SUS and detector retention. For the primary wildfire semantic-utility objective, the no-detail selector is currently stronger. Detail-aware scoring should be treated as an optional reconstruction-balanced mode rather than the default mission-utility mode.

Model improvement Step 5 applies that decision in the software. The default token selection mode is now **mission_utility**, using utility + entropy + cost without the detail term. The detail-aware selector remains available as **reconstruction_balanced** for demos or experiments that prioritize visual reconstruction quality.

Model improvement Step 6 starts the next research direction: a **mode-conditioned learned token selector**. This optional PyTorch model predicts token priorities from VQ-VAE token IDs, utility maps, entropy/detail features, and the requested mode. It is implemented and unit tested, but not yet trained or used as the default API path.

Model improvement Step 7 trained that learned selector on **100 Sentinel-2 patches for 10 epochs** using teacher distillation from the validated fixed selectors. Mean training loss decreased from **0.023312** to **0.004012**. This shows the learned selector can fit the mode-conditioned teacher signal; it still needs held-out benchmarking before it should be used in the API.

Model improvement Step 8 scaled training to **1,500 real Sentinel-2/CEMS-derived patches** with a reproducible 80/20 train/validation split. The learned selector trained on 1,200 patches and validated on 300 held-out patches. Final train loss was **0.002844** and final validation loss was **0.003845**.

Model improvement Step 9 adds the route for true mask-supervised model improvement. The repository now supports research wildfire datasets such as **CEMS-HLS**, **HLS Burn Scars**, **FireScope-Bench**, and **EO4WildFires** through a reproducible download/manifest script. This is the recommended path for improving the AI selector because it trains token priorities from real wildfire/burn-scar masks rather than only from hand-designed utility heuristics.

To preview the dataset plan:

```bash
python scripts/download_research_wildfire_datasets.py --datasets cems_hls hls_burn_scars --profile small
```

To download the controlled small profile and build the training manifest:

```bash
python scripts/download_research_wildfire_datasets.py --datasets cems_hls hls_burn_scars --profile small --execute
```

To train the mask-supervised selector after data preparation:

```bash
python scripts/train_mask_supervised_token_selector.py --manifest datasets/research_wildfire/wildfire_research_manifest.csv --epochs 8
```

Important honesty note:

> JPEG remains a very strong baseline for general image reconstruction. This project does not claim to replace JPEG everywhere. The research question is whether mission utility can be preserved efficiently under extreme satellite communication constraints.

---

## 4. How The System Works

```text
Sentinel-2 / Wildfire Image
        ↓
Wildfire Utility Detector
        ↓
Semantic Utility Map
        ↓
VQ-VAE Encoder
        ↓
Utility-Aware Token Ranking
        ↓
Adaptive Transmission
        ↓
Reconstruction
        ↓
SUS + Detector Retention Evaluation
```

Plain-English explanation:

1. The system receives an image.
2. It estimates which areas are important for wildfire monitoring.
3. It converts the image into learned tokens.
4. It ranks the tokens by mission importance.
5. It keeps the most important tokens when bandwidth is limited.
6. It reconstructs the image.
7. It measures how much wildfire information survived.

---

## 5. Repository Structure

```text
backend/          FastAPI application and compression services
frontend/         Streamlit demo dashboard
semantic_ai/      wildfire, flood, and ship utility detectors
token_selection/  utility-aware token ranking and pruning
metrics/          Semantic Utility Score and related metrics
communication/    satellite downlink and bandwidth analysis
transmission/     energy and token transmission models
evaluation/       benchmark and statistical evaluation pipelines
datasets/         dataset loader code only; raw datasets are not committed
scripts/          command-line runners for experiments and reports
tests/            automated checks for core components
reports/          proposal, experiment report, and supervisor material
results/          summary CSVs and publication-ready result tables
docs/             methodology and code walkthrough documentation
```

---

## 6. Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Model Checkpoint

The public repository does **not** include large model checkpoint files.

Expected checkpoint location:

```text
checkpoints/vqvae_s16k8.pt
```

You can also set a custom path:

```bash
set COMPRESSAI_CHECKPOINT_PATH=path\to\your\checkpoint.pt
```

On macOS/Linux:

```bash
export COMPRESSAI_CHECKPOINT_PATH=path/to/your/checkpoint.pt
```

If you only want to inspect the code, reports, and results, you do not need the checkpoint.  
If you want to run image compression or reconstruction, you need a compatible VQ-VAE checkpoint.

---

## 8. Run The API

Start the FastAPI backend:

```bash
uvicorn backend.app:app --reload
```

Open the API documentation:

```text
http://localhost:8000/docs
```

Main endpoint:

```http
POST /compress
```

Research endpoints:

```http
POST /analyze-semantic-regions
POST /simulate-transmission
POST /benchmark
```

---

## 9. Run The Demo Dashboard

Start the Streamlit frontend:

```bash
streamlit run frontend/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The dashboard lets a user upload an image and view:

- original image
- reconstructed image
- compression ratio
- bandwidth saved
- semantic token count
- mission utility metrics

---

## 10. Run Tests

Run the basic automated checks:

```bash
pytest
```

These tests check important internal pieces such as:

- image metrics
- semantic region analysis
- token pruning
- satellite transmission simulation

---

## 11. Reproduce Summary Results

The public repository includes summary CSV files under:

```text
results/summary_tables/
```

The main reports are:

```text
reports/experiment_results_2000_word_report.md
reports/phd_application_research_proposal.md
reports/final_supervisor_ready_phd_proposal.md
```

To rerun full benchmarks, you need the datasets locally. Raw datasets are not committed because they are large and may have separate licenses.

---

## 12. Research Reports

Useful documents:

- [Experiment Results Report](reports/experiment_results_2000_word_report.md)
- [Testing and Validation Report](reports/testing_validation_report.md)
- [JPEG2000 / CCSDS-Style Baseline Report](results/space_codec_baselines_500/space_codec_baseline_report.md)
- [Model Improvement Step 1: Operating Point Analysis](results/model_improvement_step1_operating_points/operating_point_analysis_report.md)
- [Model Improvement Step 2: Detail-Aware Token Scoring](results/model_improvement_step2_token_scoring/token_scoring_improvement_report.md)
- [Model Improvement Step 3: Sentinel-2 Detail-Term Ablation](results/model_improvement_step3_detail_ablation/detail_term_ablation_report.md)
- [Model Improvement Step 4: 500-Patch Detail-Term Ablation](results/model_improvement_step4_detail_ablation_500/detail_term_ablation_report.md)
- [Model Improvement Step 5: Mission-Utility Default Selector](results/model_improvement_step5_mission_utility_default/mission_utility_default_report.md)
- [Model Improvement Step 6: Mode-Conditioned Learned Token Selector](results/model_improvement_step6_mode_conditioned_selector/mode_conditioned_selector_report.md)
- [Model Improvement Step 7: Learned Selector Training](results/model_improvement_step7_learned_selector_training/training_report.md)
- [Model Improvement Step 8: Large Sentinel-2 Training](results/model_improvement_step8_large_satellite_training/training_report.md)
- [PhD Application Research Proposal](reports/phd_application_research_proposal.md)
- [Supervisor-Ready Proposal](reports/final_supervisor_ready_phd_proposal.md)
- [Code Walkthrough](docs/code_walkthrough.md)

---

## 13. Who Is This For?

This project is useful for:

- Earth Observation researchers
- PhD supervisors
- remote sensing groups
- satellite AI teams
- space-tech incubators
- wildfire monitoring researchers
- edge AI engineers
- semantic communication researchers

---

## 14. Current Limitations

The project is a research prototype, not a flight-certified satellite system.

Known limitations:

- Sentinel-2 validation currently uses a 100-scene benchmark, not yet 500+ scenes.
- JPEG remains stronger for many general reconstruction settings.
- Large datasets and checkpoints are not included in this public repo.
- Real Jetson or flight-hardware deployment still needs further validation.
- FIRMS and burned-area label alignment should be expanded.
- Detail-aware token scoring improves reconstruction metrics modestly, but slightly reduces SUS and detector retention on the 500-patch Sentinel-2 ablation.
- The learned mode-conditioned selector has completed 1,500-patch Sentinel-2 training, but still needs held-out dataset-level compression validation.

---

## 15. Roadmap

Next research steps:

- make the no-detail selector the mission-utility default and keep detail-aware scoring as an optional reconstruction-balanced mode
- train and benchmark the mode-conditioned learned token selector
- run learned-vs-fixed selector benchmarking on held-out Sentinel-2 patches
- improve FIRMS and burn-scar label alignment
- produce an arXiv preprint
- add a short demo video
- benchmark on Jetson-class hardware
- prepare IEEE/IGARSS-style workshop submission

---

## 16. Suggested Citation

```bibtex
@misc{jayakumar2026spacesemanticcompression,
  title={Semantic Utility-Aware Compression for Wildfire-Centric Earth Observation Systems},
  author={Jayakumar, Athul},
  year={2026},
  note={Research prototype for semantic communication and edge intelligence in Earth Observation}
}
```

---

## 17. Contact

Author: **Athul Jayakumar**  
Research focus: AI, semantic communication, neural compression, Earth Observation, and edge intelligence for satellites.
