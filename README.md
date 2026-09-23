# Space Semantic Compression

**Research release v0.2.0 - Semantic utility-aware communication for wildfire Earth Observation**

This repository studies one focused question:

> When a satellite cannot transmit a complete image, can it preserve wildfire-relevant information by sending learned image tokens in order of mission utility?

The project combines a VQ-VAE image representation, wildfire utility maps, token selection, measured payload serialisation, reconstruction, classical-codec baselines and event-paired evaluation. It is a research prototype, not flight software or an operational warning system.

## Current Scientific Status

The central hypothesis is **unconfirmed**. The latest frozen comparison shows that training adaptation improved the learned model, but JPEG2000 remained stronger at the tested byte ceiling.

The principal current result uses 60 EcoFireBias event pairs: 60 post-fire Sentinel-2 RGB chips and 60 matched negative chips, each at 224 x 224 pixels. Every method obeyed a maximum 1,200-byte serialised payload per image. This is a common ceiling, not exactly identical transmitted sizes.

| Method | Burn SUS | dNBR-proxy Dice | PSNR (dB) | SSIM | Detector retention | Negative predicted-positive area | Mean payload (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JPEG | 92.72 | 0.519 | 27.97 | 0.941 | 0.973 | 0.509 | 1,116.7 |
| JPEG2000 RDO | **95.12** | **0.533** | 27.18 | 0.913 | **0.985** | 0.509 | 1,177.5 |
| Base VQ-VAE | 70.85 | 0.286 | 19.87 | 0.644 | 0.728 | 0.225 | 1,140.8 |
| Adapted VQ-VAE | 83.28 | 0.469 | 20.78 | 0.636 | 0.869 | 0.509 | 1,155.4 |

Adapted minus base VQ-VAE was **+12.43 SUS** (95% event-paired bootstrap CI +8.74 to +16.31) and **+0.183 proxy Dice** (+0.122 to +0.245). Adapted minus JPEG2000 was **-11.85 SUS** (-15.14 to -8.72) and **-0.064 Dice** (-0.113 to -0.018). The adapted model also increased predicted-positive area on negative chips by 0.284 relative to the base model.

These observations establish a working platform and a clear PhD research problem. They do **not** establish superiority over JPEG2000, operational wildfire benefit or independent geographic generalisation. The dNBR masks are quantised spectral proxies, all selected validation countries occur in training, and patch event IDs do not prove source-footprint independence. The 37-pair EcoFireBias test cohort remains sealed and unscored.

Read [RESEARCH_STATUS.md](RESEARCH_STATUS.md) before citing results. The complete frozen report is [VALIDATION_REPORT.md](results/future_satellite_cohort_audit/ecofirebias_official_validation/VALIDATION_REPORT.md).

## System Pipeline

```text
Sentinel-2 or wildfire image
            |
            v
Wildfire utility estimator -> utility map
            |
            v
VQ-VAE encoder -> discrete tokens
            |
            v
Utility-aware ranking -> byte-limited serialised payload
            |
            v
VQ-VAE decoder -> reconstructed image
            |
            v
SUS components + independent proxy mask + quality and payload metrics
```

The receiver reconstructs from transmitted token codes and the transmitted selection mask only. Historical experiments that used information from the untransmitted token grid are retained as development history and are not current publication evidence.

## Release Model

The release model is identified as:

```text
model_id: vqvae-ecofirebias-adapted-2026-09
architecture: VQ-VAE, 8192 codes, 256-dimensional codes, stride 16
checkpoint: models/checkpoints/vqvae_ecofirebias_train_adapted.pt
sha256: 668cf5bda5c69b196d70306dbc6ed576c2675d372ffcfe2605e53a40286011f6
status: experimental; failed advancement gate; not approved for sealed-test scoring
```

Large checkpoint files are intentionally excluded from Git. See [MODEL_CARD.md](MODEL_CARD.md) for training provenance, limitations and correct use. The API verifies the expected SHA-256 hash when the default release checkpoint is loaded. For a custom checkpoint, set both `COMPRESSAI_CHECKPOINT_PATH` and `COMPRESSAI_CHECKPOINT_SHA256`; an empty hash disables verification for local development.

## Repository Structure

```text
backend/          FastAPI routes and services
frontend/         Streamlit research dashboard
src/models/       VQ-VAE implementation
semantic_ai/      wildfire utility and burn-scar models
token_selection/  fixed and learned token-ranking methods
evaluation/       matched-rate, label-fidelity and validation pipelines
communication/    satellite-link analysis
datasets/         dataset loaders and preparation code
scripts/          training, evaluation, audit and report commands
tests/            automated regression and protocol checks
results/          frozen summaries and experiment records
reports/          application proposal and evidence briefs
docs/             methods, diagrams and validation documentation
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Place a compatible checkpoint at the configured path before running compression. Datasets and model weights are not included because of size and source-specific licensing.

## Run the Application

FastAPI backend:

```bash
uvicorn backend.app:app --reload
```

API documentation: `http://localhost:8000/docs`

Main endpoints:

```text
GET  /health
POST /compress
POST /reconstruct
POST /benchmark
POST /simulate-transmission
POST /analyze-semantic-regions
```

Streamlit dashboard:

```bash
streamlit run frontend/streamlit_app.py
```

Dashboard: `http://localhost:8501`

## Verification and Tests

Verify the release metadata, frozen evidence and optional local checkpoint:

```bash
python scripts/verify_research_release.py
```

Run the automated tests:

```bash
pytest -q
```

The complete frozen evaluation requires separately obtained source data. The repository includes publishable hashes, plans, reports and scripts needed to inspect the protocol without redistributing licensed imagery or machine-specific path manifests.

## Research and Application Documents

- [Current PhD proposal](reports/phd_proposal_submission_2026.md)
- [Application-pack index](reports/phd_application_index_2026.md)
- [Preliminary evidence brief](career_evidence_pack/phd/phd_evidence_brief_2026.md)
- [Submission reconciliation](career_evidence_pack/phd/submission_reconciliation_2026.md)
- [Matched-rate validation notes](docs/matched_rate_validation.md)
- [Visual diagram pack](docs/diagrams/visual_diagram_pack.md)

Older proposal files and exploratory result directories remain for provenance. They are superseded where they conflict with the September 2026 frozen validation report.

## Safe Interpretation

Appropriate claim:

> Training adaptation improved the VQ-VAE on a frozen Sentinel-2 validation cohort, while JPEG2000 remained stronger at the tested 1,200-byte ceiling. The result motivates independent-label, scene-separated and exactly rate-matched PhD research.

Unsupported claims:

- the learned method beats JPEG2000;
- the validation cohort is geographically independent;
- quantised dNBR proxies are manually verified ground truth;
- the sealed test cohort has been evaluated;
- simulated communication or energy estimates are measured flight performance;
- the current software is deployment-ready for spacecraft.

## Next Scientifically Valid Improvement

The next model comparison should use a genuinely untouched, licensed Sentinel-2 cohort with independent masks, source-scene and footprint separation, positive and negative cases, and a protocol frozen before model scoring. The primary endpoint should combine SUS with independent-mask overlap and a negative-scene specificity constraint. Repeated tuning on the existing 18-event development or 60-event validation cohorts would weaken, not improve, the evidence.

The bounded implementation path is documented in [Model Improvement Plan v0.2](docs/model_improvement_plan_v0_2.md).

## Citation

See [CITATION.cff](CITATION.cff). Until a preprint is publicly archived, cite this repository as software rather than as a peer-reviewed publication.

## Author

**Athul Jayakumar**
Research focus: Earth Observation AI, neural compression, semantic communication and resource-constrained edge intelligence.
