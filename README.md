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

---

## 15. Roadmap

Next research steps:

- expand Sentinel-2 wildfire validation to 500+ scenes
- add JPEG2000 and CCSDS-style baselines
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
