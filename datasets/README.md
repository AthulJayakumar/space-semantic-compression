# Datasets

This folder contains **dataset loader code only**.

Raw datasets are intentionally not committed to GitHub because they are large and may have separate licenses.

Expected local layout for wildfire image datasets:

```text
datasets/
  dfire/
    images/
    labels/
  flame/
    images/
  sentinel2/
    images/
    metadata/
    labels/
```

Supported loaders:

- `wildfire_datasets.py`: DFire, FLAME, MODIS/FIRMS-style wildfire image loaders
- `sentinel2.py`: Sentinel-2 RGB, GeoTIFF, and metadata discovery
- `firms.py`: NASA FIRMS CSV loading and Sentinel-2 fire-mask generation

For quick code review, datasets are not required.  
For full benchmark reproduction, download the datasets separately and place them in this folder.
