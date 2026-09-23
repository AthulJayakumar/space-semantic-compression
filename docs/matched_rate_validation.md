# Matched-Rate Satellite Validation

This experiment compares JPEG, JPEG2000, random VQ-VAE token selection,
entropy-only selection, and utility-aware selection at equal transmitted-byte
budgets. It uses independently stored HLS wildfire tiles rather than treating
multiple patches from one source image as independent scenes.

## 1. Acquire the official dataset

```powershell
C:\Conda\envs\img2word\python.exe scripts\download_research_wildfire_datasets.py --datasets hls_burn_scars --profile standard --execute
```

## 2. Prepare and audit 500 tiles

```powershell
C:\Conda\envs\img2word\python.exe scripts\prepare_validated_satellite_benchmark.py --limit 500
```

The command creates a scene manifest, split audit, integrity JSON, and validity
report in `results/validated_hls_500`. Geographic HLS tile groups are assigned
to only one partition. The report explicitly distinguishes independent 512x512
HLS tiles from full satellite granules and from correlated image patches.

## 3. Run the held-out matched-rate benchmark

```powershell
C:\Conda\envs\img2word\python.exe scripts\run_matched_rate_benchmark.py --output-dir results\matched_rate_hls_500
```

The run evaluates the geography-held-out test split. Its primary confirmatory
tables use the intersection with the provider's original validation split.

## 4. Generate the evidence report

```powershell
C:\Conda\envs\img2word\python.exe scripts\generate_matched_rate_evidence_report.py
```

Read `results/matched_rate_hls_500/publication_evidence_report.md` together with
`results/validated_hls_500/dataset_validity_report.md`. Do not describe the
current dataset as full Sentinel-2 scenes, event-separated, smoke-labelled, or
cloud-ground-truthed; the generated validity report records these limitations.
