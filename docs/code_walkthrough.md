# Code Walkthrough

This document explains the repository in plain English.

## Big Picture

The project answers one question:

> If a satellite cannot send a full image, can it send the most mission-important image tokens first?

For this prototype, the mission is wildfire monitoring.

## Main Flow

1. `backend/app.py` starts the API.
2. `backend/routes/compress.py` receives an uploaded image.
3. `backend/services/compression_service.py` coordinates the full pipeline.
4. `semantic_ai/wildfire_detector.py` estimates wildfire importance.
5. `src/models/vqvae.py` encodes the image into learned tokens.
6. `token_selection/utility_pruner.py` keeps the most useful tokens.
7. `backend/services/decoder_service.py` reconstructs the image.
8. `metrics/semantic_utility.py` measures how much mission utility survived.
9. `communication/satellite_analysis.py` estimates downlink savings.

## Important Modules

### `semantic_ai/`

Mission detectors live here. They transform an image into a **utility map**.

A utility map is a heatmap. Bright areas mean:

> This part of the image is important for the mission.

For wildfire, the detector looks for fire-like, smoke-like, and burn-scar-like evidence.

### `token_selection/`

This is where the system decides what to transmit.

The most important file is:

```text
token_selection/utility_pruner.py
```

It combines:

- semantic utility
- token entropy
- communication cost

and returns a mask saying which tokens should be kept.

### `metrics/`

This folder contains the research metric:

```text
Semantic Utility Score (SUS)
```

SUS measures mission preservation, not just visual quality.

### `evaluation/`

This folder runs experiments across datasets and baselines.

It produces:

- CSV tables
- statistical summaries
- retention studies
- publication-style results

### `communication/` and `transmission/`

These modules connect compression to satellite communication.

They estimate:

- bandwidth saved
- downlink time saved
- data volume reduction
- approximate energy impact

### `frontend/`

The Streamlit app gives a simple visual demo for people who do not want to use the command line.

## What To Read First

For a non-technical reviewer:

1. `README.md`
2. `reports/experiment_results_2000_word_report.md`
3. `reports/phd_application_research_proposal.md`
4. `docs/code_walkthrough.md`

For a technical reviewer:

1. `backend/services/compression_service.py`
2. `semantic_ai/wildfire_detector.py`
3. `token_selection/utility_pruner.py`
4. `metrics/semantic_utility.py`
5. `evaluation/wildfire_validation.py`
6. `evaluation/sentinel2_validation.py`
