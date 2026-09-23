"""Record the predeclared training-only fine-tune and its exploratory gate reuse."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decide_ecofirebias_development_gate import evaluate_run, paired_result  # noqa: E402

AUDIT = ROOT / "results/future_satellite_cohort_audit"
OUT = AUDIT / "ecofirebias_training_expansion"
BEFORE = AUDIT / "development_mixed_wire_1200/extreme_rate_rows.csv"
AFTER = OUT / "development_adapted_1200/extreme_rate_rows.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def report() -> dict[str, object]:
    plan_path = OUT / "TRAINING_PLAN_FROZEN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    data_audit = json.loads((OUT / "TRAINING_DATA_AUDIT.json").read_text(encoding="utf-8"))
    if data_audit["training_plan_sha256"] != sha256(plan_path) or data_audit["train_validation_event_overlap"]:
        raise ValueError("Training data or event split differs from the frozen plan")
    checkpoint_path = ROOT / plan["checkpoint_output"]
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    recorded = saved["fine_tuning"]
    for key, expected in plan["trainer_arguments"].items():
        if key == "semantic_detector_checkpoint":
            actual = recorded[key].replace("\\", "/")
        else:
            actual = recorded[key]
        if actual != expected:
            raise ValueError(f"Checkpoint setting {key} differs from frozen plan: {actual} != {expected}")
    if sha256(Path(plan["base_checkpoint"])) != plan["base_checkpoint_sha256"]:
        raise ValueError("Base checkpoint changed after freezing")
    if recorded["train_samples"] != 480 or recorded["validation_samples"] != 120:
        raise ValueError("Checkpoint sample counts do not match predeclared split")

    gate = json.loads((AUDIT / "DEVELOPMENT_GATE_FROZEN.json").read_text(encoding="utf-8"))
    metadata = {row["example_id"]: row for row in read_csv(ROOT / "datasets/wildfire_global_v1/metadata.csv")}
    before_rows, after_rows = read_csv(BEFORE), read_csv(AFTER)
    before = evaluate_run(before_rows, gate, metadata)
    after = evaluate_run(after_rows, gate, metadata)
    before_vq = {row["sample_id"]: row for row in before_rows if row["method"] == "vqvae_fixed_utility"}
    after_vq = {row["sample_id"]: row for row in after_rows if row["method"] == "vqvae_fixed_utility"}
    before_jp2 = {row["sample_id"]: row for row in before_rows if row["method"] == "jpeg2000_rdo"}
    after_jp2 = {row["sample_id"]: row for row in after_rows if row["method"] == "jpeg2000_rdo"}
    for sample_id in gate["development_ids"]:
        if before_jp2[sample_id] != after_jp2[sample_id]:
            raise ValueError("Classical baseline changed between development runs")
    positive = [sample_id for sample_id in gate["development_ids"] if metadata[sample_id]["kind"] == "burn"]
    negative = [sample_id for sample_id in gate["development_ids"] if metadata[sample_id]["kind"] == "neg"]
    changes = {
        metric: paired_result(
            [float(after_vq[sample_id][metric]) - float(before_vq[sample_id][metric]) for sample_id in ids],
            gate["gate"],
        )
        for metric, ids in (
            ("sus", positive), ("label_dice", positive), ("psnr", positive),
            ("ssim", positive), ("detector_retention", positive),
            ("predicted_positive_fraction", negative),
        )
    }
    result = {
        "status": "training_only_adaptation_exploratory_development_reuse",
        "frozen_training_plan_sha256": sha256(plan_path),
        "frozen_development_gate_sha256": sha256(AUDIT / "DEVELOPMENT_GATE_FROZEN.json"),
        "training_manifest_sha256": data_audit["training_manifest_sha256"],
        "base_checkpoint_sha256": plan["base_checkpoint_sha256"],
        "adapted_checkpoint_sha256": sha256(checkpoint_path),
        "before_rows_sha256": sha256(BEFORE),
        "after_rows_sha256": sha256(AFTER),
        "training_events": 480,
        "internal_validation_events": 120,
        "development_events": 18,
        "sealed_test_scored": False,
        "before": before,
        "after": after,
        "paired_adapted_minus_base": changes,
        "interpretation": "The development set has been consulted twice; improvement versus base is exploratory and cannot establish independent superiority",
    }
    (OUT / "ADAPTATION_DECISION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    jp2 = after["summaries"]["jpeg2000_rdo"]
    base = before["summaries"]["vqvae_fixed_utility"]
    adapted = after["summaries"]["vqvae_fixed_utility"]
    lines = [
        "# EcoFireBias training-only adaptation", "",
        "**Decision: no-go for sealed testing.** The 18-event development set was reused after a predeclared two-epoch fine-tune. Its results are exploratory.", "",
        "| Method | Burn SUS | Burn dNBR-proxy Dice | Burn PSNR | Mean serialized bytes |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name, summary in (("JPEG2000 RDO", jp2), ("Mixed-wire base VQ-VAE", base), ("Adapted VQ-VAE", adapted)):
        lines.append(f"| {name} | {summary['mean_positive_sus']:.2f} | {summary['mean_positive_proxy_dice']:.3f} | {summary['mean_positive_psnr']:.2f} | {summary['mean_encoded_bytes']:.1f} |")
    lines.extend(["", "## Paired adapted-minus-base changes", ""])
    for metric, comparison in changes.items():
        lines.append(f"- {metric}: {comparison['mean_difference']:+.3f}; 95% paired-event bootstrap CI [{comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}].")
    lines.extend(["", "The frozen gate still fails: " + ", ".join(key for key, passed in after["checks"].items() if not passed) + ".", "",
                  "Training used 600 distinct EcoFireBias train events with a 480/120 event-disjoint internal split. No gate-development, label-calibration, or test event entered this training manifest. The detector and token selector were unchanged. The development set is not independent after repeated consultation, the dNBR mask is a quantized proxy, and no sealed test image was scored.", ""])
    (OUT / "ADAPTATION_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    result = report()
    print(json.dumps({"adapted_gate_passed": result["after"]["advance"], "sealed_test_scored": False}, indent=2))


if __name__ == "__main__":
    main()
