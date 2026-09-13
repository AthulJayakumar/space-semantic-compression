"""Tests for large wildfire research dataset manifest utilities."""

from pathlib import Path

from PIL import Image

from datasets.research_wildfire import build_manifest, dataset_plan, validate_manifest


def test_dataset_plan_contains_recommended_large_sources():
    plan = dataset_plan(["cems_hls", "hls_burn_scars"])

    names = {item["name"] for item in plan}

    assert names == {"cems_hls", "hls_burn_scars"}
    assert all(item["repo_id"] for item in plan)


def test_build_manifest_pairs_images_and_masks(tmp_path: Path):
    image_dir = tmp_path / "images" / "train"
    mask_dir = tmp_path / "masks" / "train"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(120, 80, 40)).save(image_dir / "scene_001.png")
    Image.new("L", (16, 16), color=255).save(mask_dir / "scene_001_mask.png")

    rows = build_manifest(tmp_path, "toy_fire")
    integrity = validate_manifest(rows)

    assert len(rows) == 1
    assert rows[0].dataset == "toy_fire"
    assert rows[0].split == "train"
    assert rows[0].width == 16
    assert rows[0].height == 16
    assert integrity["valid"] is True
