"""Frozen-gate decision and invalid-edge masking checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from evaluation.matched_rate import BenchmarkItem, EncodedCandidate, MatchedRateBenchmark
from scripts.decide_ecofirebias_development_gate import evaluate_run, paired_bootstrap_ci


def synthetic_gate_rows() -> tuple[list[dict[str, str]], dict[str, object], dict[str, dict[str, str]]]:
    ids = [f"{event}_{kind}" for event in range(18) for kind in ("burn", "neg")]
    metadata = {sample_id: {"event_id": sample_id.split("_")[0], "kind": sample_id.split("_")[1]} for sample_id in ids}
    gate = {
        "development_ids": ids,
        "maximum_wire_bytes_per_image": 1200,
        "gate": {
            "burn_positive_sus_mean_difference_at_least": 3.0,
            "burn_positive_proxy_dice_mean_difference_at_least": 0.02,
            "paired_event_bootstrap_95_percent_lower_bound_for_both_differences_above": 0.0,
            "negative_pair_false_positive_area_mean_increase_at_most": 0.02,
            "bootstrap_seed": 4,
            "bootstrap_resamples": 100,
        },
    }
    rows = []
    for sample_id in ids:
        for method in ("vqvae_fixed_utility", "jpeg_rdo", "jpeg2000_rdo"):
            candidate = method == "vqvae_fixed_utility"
            rows.append({
                "sample_id": sample_id,
                "event_group": metadata[sample_id]["event_id"],
                "source_split": "train",
                "split": "development",
                "method": method,
                "target_bytes": "1200",
                "encoded_bytes": "1100",
                "sus": str(85 + 0.1 * int(metadata[sample_id]["event_id"])) if candidate else "80",
                "label_dice": str(0.5 + 0.001 * int(metadata[sample_id]["event_id"])) if candidate else "0.4",
                "psnr": "30",
                "ssim": "0.9",
                "detector_retention": "0.8",
                "predicted_positive_fraction": str(0.05 + 0.001 * int(metadata[sample_id]["event_id"])) if candidate else "0.1",
            })
    return rows, gate, metadata


def test_bootstrap_is_deterministic() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert paired_bootstrap_ci(values, 100, 4) == paired_bootstrap_ci(values, 100, 4)


def test_joint_gate_passes_only_when_all_conditions_hold() -> None:
    rows, gate, metadata = synthetic_gate_rows()
    assert evaluate_run(rows, gate, metadata)["advance"]
    for row in rows:
        if row["method"] == "vqvae_fixed_utility" and metadata[row["sample_id"]]["kind"] == "burn":
            row["label_dice"] = str(0.3 + 0.001 * int(metadata[row["sample_id"]]["event_id"]))
    result = evaluate_run(rows, gate, metadata)
    assert not result["advance"]
    assert not result["checks"]["proxy_dice_mean_advantage"]


def test_gate_rejects_changed_byte_ceiling() -> None:
    rows, gate, metadata = synthetic_gate_rows()
    rows[0]["target_bytes"] = "1300"
    with pytest.raises(ValueError, match="ceiling"):
        evaluate_run(rows, gate, metadata)


def test_invalid_edge_pixels_are_excluded_from_label_metrics(tmp_path: Path) -> None:
    truth = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    valid = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    truth_path, valid_path = tmp_path / "truth.png", tmp_path / "valid.png"
    Image.fromarray(truth).save(truth_path)
    Image.fromarray(valid).save(valid_path)
    result = SimpleNamespace(confidence_map=np.array([[1.0, 1.0], [0.0, 0.0]]), backend="test")
    detector = SimpleNamespace(detect=lambda _: result, _supervised_config={"threshold": 0.5})
    components = SimpleNamespace(detector_retention=1.0, object_retention=1.0, relevance_retention=1.0, region_preservation=1.0)
    runner = object.__new__(MatchedRateBenchmark)
    runner._truth_masks = {}
    runner.skip_lpips = True
    runner.service = SimpleNamespace(
        detectors={"wildfire_detection": detector},
        _detect_mission_utility=lambda *_: result,
        semantic_utility_metric=SimpleNamespace(score_before_after=lambda *_: (100.0, components)),
        metrics_service=SimpleNamespace(psnr=lambda *_: 30.0, ssim=lambda *_: 0.9),
    )
    original = Image.new("RGB", (2, 2))
    item = BenchmarkItem("sample", tmp_path / "image.png", mask_path=truth_path, valid_mask_path=valid_path)
    candidate = EncodedCandidate(b"x", original, 1.0)
    row = runner._measure(item, original, result, (1, 1), "jpeg", 0, 0.1, 10, 1, 10, candidate)
    assert row["label_dice"] == 1.0
    assert row["label_false_positive_rate"] == 0.0
    assert row["label_valid_fraction"] == 0.75
