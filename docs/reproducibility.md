# Reproducibility Notes

## What Is Reproducible From This Repository?

The repository includes:

- source code
- tests
- dataset loader code
- benchmark runners
- summary result tables
- research reports
- methodology figures

The repository does not include:

- raw DFire images
- raw FLAME images
- raw Sentinel-2/CEMS imagery
- large model checkpoints
- generated heatmap dumps

Those files are intentionally excluded to keep the GitHub repository clean and legally safer.

## Basic Verification

Run:

```bash
pytest
```

This verifies that the core service and research components still import and behave correctly.

## Full Benchmark Reproduction

To rerun full experiments:

1. Download datasets into `datasets/`.
2. Place a compatible VQ-VAE checkpoint at `checkpoints/vqvae_s16k8.pt`.
3. Run the relevant script:

```bash
python scripts/run_wildfire_validation.py
python scripts/run_sentinel2_validation.py
```

Outputs are written to `results/`.

## Results Included In This Public Repo

The public repo includes summary tables in:

```text
results/summary_tables/
```

These tables are the compact evidence used in the research reports.
