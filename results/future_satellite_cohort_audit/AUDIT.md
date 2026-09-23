# Prospective satellite validation: local-data audit

**Status: a geographically screened EcoFireBias cohort is staged, but remains
sealed.** This is a data, lineage and split audit, not a model evaluation.
No reserved images were scored.

| Source | Available metadata | Previously used or reserved | Fresh-test limitation |
|:--|:--|:--|:--|
| HLS burn scars | 804 archived image tiles | 464 used in mixed training | The other 340 provide zero new geographic groups outside known training/holdout groups; event IDs unavailable. |
| CEMS/HLS | 439 tiles: 307 train, 52 validation, 80 test | 307 used in mixed training | GeoTIFF metadata reveal 121 Copernicus event codes and projected locations; 34 codes occur in both source train and test. |
| ImpactMesh-Fire | 1,270 local test tiles, 43 AOI events | Prior 100-tile evaluation touched all 43 AOI events | Remaining tiles are not an untouched event-level replication set; 45 evaluated tiles share 19 codes with the newest mixed model's CEMS training, and 51 share 21 codes with the full CEMS source. |
| EcoFireBias | 906 test chips, 453 paired events | 60 evaluated, 60 frozen but unevaluated | Conservative country and 100 km CEMS-source-center exclusions leave 37 untouched event pairs (74 chips). All 74 RGB and byte-valued dNBR GeoTIFFs are now staged and hash-checked, but not scored. |

The mixed-training manifest also contains 60 HLS archive tiles whose
original source partition was named `validation`. They were admitted to
training after excluding the current HLS geographic holdouts; the archive's
original validation partition is therefore **not** an untouched test for
this checkpoint.

The [EcoFireBias dataset documentation](https://huggingface.co/datasets/moritzrengert1/wildfire_global/blob/main/README.md)
describes event-level train/test splitting, paired fire and nearby negative
chips, post-fire Sentinel-2 RGB, and dNBR products. A thresholded dNBR map
could serve as an **external spectral proxy** for burn severity. It is not a
manually verified burn-scar polygon or an active-fire/smoke label. Its file
encoding, spatial alignment, invalid pixels, and threshold must be checked
before use; HLS mask Dice and dNBR-proxy Dice must not be pooled as if they
were the same endpoint.

The existing frozen EcoFireBias replication selection is distinct from the
60 events used in the prior EcoFireBias evaluation. Its original 60 pairs
include countries represented in CEMS training. Excluding those countries,
the US HLS study region, and any event within 100 km of **any** CEMS source
tile center leaves 37 pairs across five continents. This is a conservative
screen, not proof that all inherited training imagery is globally disjoint:
one ancestor's original 439-row manifest is missing. No new lockbox was
created; the pre-existing frozen selection was screened before evaluation.

## Cross-source overlap correction

The CEMS source files are not anonymous: their GeoTIFF `GDAL_METADATA` embeds
an `EMSR` event code, country and projected location. The source train/test
tile split is **not event-disjoint**: 34 Copernicus event codes appear in
both. The previously evaluated 100-tile ImpactMesh cohort contains 45 tiles
from 19 event codes also represented in the newest mixed model's CEMS training.
This does not establish duplicate pixels, but it invalidates a claim that
the entire cohort was event-disjoint from that training. Across the *full*
CEMS source, 51 ImpactMesh tiles share 21 EMSR codes. Earlier checkpoints
used different CEMS-related inputs; their exact per-checkpoint overlap is
not fully certified by this source-level count. The newest model also
inherits ancestor checkpoints that record random training on the full CEMS
manifest, so the 45-tile overlap with the newest run's 307 CEMS training
tiles is a lower bound on possible event exposure. The remaining 55 tiles
were also already evaluated, so they are not a fresh sealed
replication set for another model iteration. See `cems_event_audit.json` and
`cems_georeference.csv` for the reproducible code-level audit.

## September 22 lineage and label follow-up

Replaying the seeds, manifests and patch provenance recorded in the current
checkpoint chain finds **52 of the previously evaluated 100 ImpactMesh tiles
sharing 22 event codes** with known direct or patch training inputs. This
supersedes the 45-tile count for the newest direct CEMS training run and the
51-tile count for the full CEMS source. One additional shared code (`EMSR401`)
appears in the Sentinel-2 patch training provenance. The replay is
**conditional**: checkpoint files did not store source-manifest hashes, and
the earliest 439-row direct manifest is now empty. Therefore 52 is a known
reconstructable overlap, not a complete lineage certification. The 59-tile
HLS internal development cohort has already been consulted repeatedly and
must not be represented as independent confirmation.

The [dataset card](https://huggingface.co/datasets/moritzrengert1/wildfire_global/blob/39f331e50458fba3669837d69fc2fdddbb1d1d69/README.md)
describes continuous dNBR, but all 74 staged "raw dNBR" GeoTIFFs are
224x224 `uint8` images. Treating those bytes as physical dNBR would be a
serious label error. On **36 separate training-split chips** from 18 events,
the rule `byte > 85` approximated the metadata's dNBR > 0.27 positive-pixel
fractions with mean absolute error 0.00097 and maximum 0.00477. This rule is
fixed as a **quantized spectral proxy**, not an exact physical dNBR inversion
or a manually checked fire boundary. Spatial correspondence between source
RGB and dNBR pixels has not been independently verified from original
Sentinel-2 bands.

Before model evaluation, an 18-event/36-chip *train-split development screen*
was frozen separately from the 18 events used for label-format checking.
The native 224x224 wire ceiling is **1,200 bytes per image**, including all
payload overhead. This exceeds the largest minimum JPEG payload (1,006
bytes) in the previously observed 120-chip development run; it is a
feasibility choice, not evidence of performance. Advancement requires, on
the paired development events at the same byte ceiling, a mean SUS advantage
of at least 3 points and mean positive-chip proxy-Dice advantage of at least
0.02 over JPEG2000 RDO, positive 95% paired-event bootstrap lower bounds for
both differences, and no more than 0.02 mean increase in negative-chip false
positive area. These thresholds are predeclared in
`DEVELOPMENT_GATE_FROZEN.json`; if any fail, the 37 reserved pairs remain
unscored. The prior 120-chip EcoFireBias matched-rate screen favored
JPEG2000 over the then-current VQ-VAE, so passing is **not expected without
demonstrated improvement**.

Reproduce the non-model audits with `scripts/audit_checkpoint_event_lineage.py`,
`scripts/audit_ecofirebias_replication.py`,
`scripts/fetch_ecofirebias_replication_assets.py`, and
`scripts/verify_ecofirebias_dnbr_encoding.py`. Full hashes and exclusions are
in `checkpoint_lineage.json`, `ecofirebias_replication_audit.json`,
`ecofirebias_assets_eligible.json`, and `dnbr_encoding_train_only.json`.

## Frozen development-gate outcome

The separately frozen 18-event/36-chip **training-split** development set
was downloaded and checked before model scoring. RGB PNGs matched the
post-fire GeoTIFF pixels exactly. Thirty-four RGB/dNBR grids matched exactly;
two pairs crossed UTM-zone boundaries but shared a geographic center. Their
dNBR labels were reprojected onto the RGB grid, with 96.4% coverage and
uncovered edge pixels excluded from proxy-mask statistics. The byte > 85
proxy fraction differed from the metadata's dNBR > 0.27 fraction by at most
0.00676 on these 36 source grids. Original continuous dNBR and all invalid
pixel locations remain unavailable from the 8-bit product.

Both existing wire-decoded VQ-VAE checkpoints were screened at the frozen
**1,200-byte per-image ceiling**, using full serialized token payload sizes.
Classical JPEG/JPEG2000 search included the native 224-pixel size, and selected
the best original-image PSNR setting under the same ceiling. Across 18 burn
chips, JPEG2000 RDO scored **91.99 mean SUS** and **0.455 mean dNBR-proxy
Dice**. The mixed-wire VQ-VAE scored **70.01 SUS / 0.243 proxy Dice** and the
mask-aware VQ-VAE **64.42 SUS / 0.162 proxy Dice**. Paired-event 95% bootstrap
intervals for both VQ-VAE-minus-JPEG2000 metrics were below zero. Mean
serialized bytes were 1,169 for JPEG2000 and 1,124 for each VQ-VAE run;
this is a common **ceiling**, not exactly identical transmitted sizes.
The predeclared joint gate failed for both checkpoints. The 37-pair sealed
cohort **was not scored**. See `DEVELOPMENT_DECISION.md` and
`development_gate_decision.json` for all five gate checks, paired intervals,
and run hashes. These are negative development findings, not an independent
test or a publication-grade comparison.

## Training-only adaptation, exploratory follow-up

A new 600-chip image-only EcoFireBias training selection was frozen before
fine-tuning: 100 event-distinct chips per continent, evenly split between
burn and negative scenes. All selected examples came from the dataset's
`train` split; none belonged to the 18 gate-development events, 18
label-encoding events, or any official test event. The existing trainer's
random image split was event-disjoint **because exactly one chip per event
was selected**: 480 training events and 120 internal-validation events.
The mixed-wire VQ-VAE was fine-tuned for two epochs without an architecture,
detector, or selector change; the base checkpoint was preserved.

On **reuse** of the same 18-event development screen at the unchanged
1,200-byte ceiling, the adapted model improved over its base from 70.01 to
85.03 burn SUS and from 0.243 to 0.417 dNBR-proxy Dice. Paired adapted-minus-
base 95% event-bootstrap intervals were [+7.91, +22.74] SUS points and
[+0.078, +0.275] proxy Dice. This is an encouraging *exploratory*
training effect, not independent confirmation: JPEG2000 still scored 91.99
SUS and 0.455 proxy Dice, the adapted-minus-JPEG2000 SUS interval remained
negative, and the frozen joint gate still returned **no-go**. Mean serialized
payload was 1,134 bytes for adapted VQ-VAE versus 1,169 for JPEG2000, both
under the common 1,200-byte ceiling. Repeatedly screening new variants on
these same events would increase selection bias. The sealed 37-pair test set
remains unscored. Full settings, hashes and paired statistics are in
`ecofirebias_training_expansion/TRAINING_PLAN_FROZEN.json` and
`ecofirebias_training_expansion/ADAPTATION_DECISION.md`.

## One official-validation generalization check

Before model scoring, 60 event pairs (120 chips, 10 pairs per continent)
were frozen from the untouched EcoFireBias `val` split. They have no event-ID
overlap with the 600 selected training events, label-calibration events,
reused development events, or sealed test events. All 60 selected validation
events are nevertheless in countries represented in selected training, and
patch-derived event IDs alone do not prove scene-footprint independence.
Source RGB, quantized dNBR labels, and spatial alignment were checked first;
five images required grid reprojection. The comparison was then run once at
the already fixed native 224 x 224 / 1,200-byte ceiling.

Across 60 burn chips, adapted VQ-VAE scored **83.28 SUS / 0.469 proxy Dice**,
compared with **70.85 / 0.286** for base VQ-VAE and **95.12 / 0.533** for
JPEG2000 RDO. Paired adapted-minus-base 95% event-bootstrap intervals were
[+8.74, +16.31] SUS points and [+0.122, +0.245] proxy Dice; adapted-minus-
JPEG2000 intervals were [-15.14, -8.72] and [-0.113, -0.018], respectively.
On 60 negative chips, adapted predicted-positive area was 0.509 versus 0.225
for base, an increase of +0.284 [95% CI +0.202, +0.365]. Adaptation thus
improved positive-chip retention but also substantially increased negative-chip
predicted-positive area. Mean serialized bytes were 1,155 adapted versus
1,178 JPEG2000, under the same ceiling but not exactly matched rates.

This single official-validation check **does not overturn the failed frozen
development gate** or authorize sealed testing. Quantized dNBR is a spectral
proxy, countries overlap, and no manual burn-scar truth or LPIPS was used.
Read `ecofirebias_official_validation/VALIDATION_PLAN_FROZEN.json` and
`ecofirebias_official_validation/VALIDATION_REPORT.md` for selection and full
paired results. No publication-grade superiority claim is supported.

## External-source metadata screen

The 73-scene Satellite Burned Area Dataset has 27 Copernicus event codes
with no exact code overlap in the local CEMS inventory, but 60 scenes lie
within 100 km of a CEMS source tile center. Only seven scenes across four
event codes pass both provisional CEMS and selected EcoFireBias-training
center-distance screens. These are not footprint checks. Licensing and raster
label integrity remain unverified, so **no external model scoring is
authorized**. A follow-up FLOGA v1 annotation screen found 21/347
year-event IDs passing provisional 100 km buffers from known CEMS and
consulted EcoFireBias footprints, but the correspondence to selectively
obtainable v2 imagery and complete inherited training geography is not
verified. FireSR also has large acquisition and modality constraints. See
`external_source_screen/SOURCE_SCREEN.md` for the source-specific limits.

## Advancement condition

1. Exclude or separately report CEMS/ImpactMesh shared events and prevent
   event-code overlap in every new training/test split.
2. Keep the 37 eligible EcoFireBias event pairs sealed. Their post-fire RGB
   and quantized dNBR files are staged; finish a training-only check of
   pixelwise alignment and fire/negative behavior before calling the proxy
   an independent segmentation endpoint.
3. The frozen 1,200-byte joint gate on the separate training-split
   development selection returned **no-go** for both existing checkpoints.
   Keep its thresholds and the 37-pair selection unchanged; do not promote
   the repeatedly consulted 59 HLS tiles to confirmatory status.
4. The training-only adaptation improved over base on the once-scored
   official validation split, but remained below JPEG2000 on SUS and proxy
   Dice and increased negative predicted-positive area. This comparison was
   event-distinct but not geography-separated. Do not tune against either
   the reused 18 events or the now-scored 60 validation events; do not score
   the sealed selection. A genuinely independent source is needed for a
   credible advancement decision.

Exact machine-readable counts are in `audit.json`; reproduce them with
`scripts/audit_future_satellite_cohort.py` and
`scripts/audit_cems_georeference.py`.
