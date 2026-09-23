# External Sentinel-2 source screen (metadata only)

No candidate imagery or model output was examined. The frozen EcoFireBias
37-pair test cohort remains sealed, and the failed development gate remains
unchanged.

## Satellite Burned Area Dataset

The [Zenodo source](https://zenodo.org/records/6597139) publishes 73
Sentinel-2/Sentinel-1 acquisitions with `satellite_data.csv` containing
Copernicus emergency event identifiers, bounds and centers. The downloaded
18.3 KB metadata file matches the source MD5
`8a93ab8cd09a4c9b7a3096bb49379ba4`. There are 27 unique event codes,
none exactly matching those in the local CEMS inventory. However, 60/73
acquisitions are within 100 km of a known CEMS source tile center, 51/73
within 100 km of a selected EcoFireBias training patch center, and only
7 acquisitions across 4 event codes pass both *center-distance* screens.
These are conservative proximity flags, not scene-footprint checks or proof
of event independence. The source itself derives its labels from Copernicus
grading maps; they are not a separate human burn-scar annotation process.
The Zenodo page does not clearly state a reuse license, and no raster or
label files were acquired, so encoding and alignment remain unknown.
**Decision: not eligible for model scoring or a confirmatory claim.**

## Other external options

- [FLOGA](https://github.com/Orion-AI-Lab/FLOGA) reports 326 Greek events
  with Hellenic Fire Service mappings and a CC BY 4.0 data license. Its
  [GeoTIFF distribution](https://huggingface.co/datasets/orion-ai-lab/FLOGA-GeoTIFFs)
  is about 1.23 TB, packaged in approximately 20 GB archives. This is a
  stronger independent-label candidate. The v1 annotation locations are
  screened below, but v2 imagery correspondence, full training lineage,
  band handling, and selective acquisition remain unverified. Do not score it.
- [FireSR](https://zenodo.org/records/13384289) offers post-fire Sentinel-2
  imagery and Canadian NBAC masks under CC BY 4.0. Its single archive is
  73.4 GB, and the documented Sentinel-2 channels are B12/B8/B4 rather
  than the current RGB input. Geographic separation and channel handling
  need an a priori audit. Do not score it yet.

The audit code and per-scene center distances are in
`scripts/screen_satellite_burned_area_source.py` and
`SOURCE_METADATA_SCREEN.json`. The next legitimate step is a feasible
licensed source with event-level metadata, exact footprint checks against
known training/selection data, verified post-fire RGB and independent
mask alignment, followed by a frozen cohort and endpoints **before** any
model evaluation. The incomplete ancestor manifest limits what can be
certified even then.

## FLOGA annotation follow-up

The public [FLOGA v1 annotation repository](https://github.com/Orion-AI-Lab/FLOGA-annotations)
was pinned at revision `f695e39964bae5292e907bc86d61c019c0e271c1`.
Its 2017-2021 polygon files contain **467 polygons and 347 distinct
year/event IDs**, not necessarily 347 available imagery pairs. A conservative
geographic screen compared each event's union bounding box with all 439 known
CEMS source tile footprints and 912 EcoFireBias samples used for training,
label calibration, development, official validation, or earlier evaluation.
The screen uses a lower bound on footprint distance and a 100 km buffer:
**325/347** events lie within that bound of a known CEMS footprint;
**123/347** lie within it of a consulted EcoFireBias chip; **21/347** pass
both provisional filters. One FLOGA event also has an exact Sentinel-2
acquisition key match to a consulted EcoFireBias sample. These counts are
not a frozen test selection and are not proof of independence: an ancestor
checkpoint manifest is missing, some other source footprints are unknown,
and v1 polygons have not been matched to v2 GeoTIFF availability.

The [FLOGA research configuration](https://github.com/Orion-AI-Lab/FLOGA/blob/main/configs/config.json)
lists Sentinel-2 B02, B03 and B04 among its bands, so an RGB composite is
plausible, but the published 20 m representation, scaling, channel order,
cloud handling, and polygon-to-raster alignment have not been verified on
any source pixels. The v2 distribution's approximately 20 GB TAR chunks do
not yet provide a credible selective acquisition path; the v1 Dropbox
folder has not been verified as offering per-event files. **No imagery or
pixel masks were downloaded or scored.** Full hashes and per-event metadata
are in `FLOGA_V1_ANNOTATION_SCREEN.json`; rerun with
`scripts/screen_floga_annotations.py` after installing `pyshp==2.3.1`.
