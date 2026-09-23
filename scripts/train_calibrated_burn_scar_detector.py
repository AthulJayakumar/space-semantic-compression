"""Train a matched baseline and calibrated burn-scar detector candidate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from evaluation.matched_rate import bootstrap_mean_ci  # noqa: E402
from evaluation.statistics import StatisticalValidator  # noqa: E402
from semantic_ai.burn_scar_model import BurnScarUtilityNet, dice_loss  # noqa: E402
from semantic_ai.burn_scar_training import (  # noqa: E402
    expected_calibration_error,
    focal_tversky_hard_negative_loss,
    probabilities_from_logits,
    select_temperature,
)


class GeographicBurnScarDataset(Dataset):
    def __init__(self, rows, image_size: int, augmentation: str = "none") -> None:
        self.rows = rows
        self.image_size = image_size
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_rgb_image(Path(row["image_path"])).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        mask = load_grayscale_image(Path(row["mask_path"])).resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        image_array = np.asarray(image, dtype="float32") / 255.0
        mask_array = (np.asarray(mask, dtype="float32") > 0.0).astype("float32")
        if self.augmentation != "none":
            image_array, mask_array = spatial_augmentation(image_array, mask_array)
        if self.augmentation == "strong":
            image_array = radiometric_augmentation(image_array)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_array.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_array[None]))
        return image_tensor, mask_tensor, row["sample_id"], row["geographic_group"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare calibrated burn-scar detector recipes.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/token_sus_selector/internal_geographic_split.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/calibrated_burn_scar_detector"))
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path("models/checkpoints/wildfire_utility_segmentation_calibrated.pt"),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"),
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    rows = read_csv(args.manifest)
    train_rows = [row for row in rows if row["split"] == "selector_train"]
    validation_rows = [row for row in rows if row["split"] == "selector_validation"]
    if args.limit is not None:
        train_rows = train_rows[: args.limit]
        validation_rows = validation_rows[: max(1, int(round(args.limit * 0.25)))]
    assert_geographic_disjoint(train_rows, validation_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.candidate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    initialization = BurnScarUtilityNet(args.base_channels).state_dict()
    started = time.perf_counter()
    baseline = train_recipe("matched_baseline", train_rows, validation_rows, initialization, args, device)
    torch.save(baseline["checkpoint"], args.baseline_checkpoint)
    if args.baseline_only:
        result = {
            "decision": "retrained_baseline_ready_for_independent_comparison",
            "selection_partition": "59-scene internal geographic validation derived from original training partition",
            "official_validation_used": False,
            "test_partition_used": False,
            "checkpoint": str(args.baseline_checkpoint),
            "metrics": baseline["aggregate"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_csv(args.output_dir / "retrained_baseline_per_scene.csv", baseline["per_scene"])
        write_csv(args.output_dir / "retrained_baseline_history.csv", baseline["history"])
        (args.output_dir / "retrained_baseline_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.output_dir / "retrained_baseline_report.md").write_text(build_baseline_report(result), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    candidate = train_recipe("calibrated_candidate", train_rows, validation_rows, initialization, args, device)
    comparison = paired_comparison(baseline["per_scene"], candidate["per_scene"], args.seed)
    go = (
        float(comparison["dice_difference_ci_low"]) > 0.0
        and float(candidate["aggregate"]["ece"]) <= float(baseline["aggregate"]["ece"])
    )
    decision = {
        "decision": "go_for_official_detector_validation" if go else "no_go_for_official_detector_validation",
        "selection_partition": "59-scene internal geographic validation derived from original training partition",
        "official_validation_used": False,
        "test_partition_used": False,
        "go_rule": "paired Dice bootstrap CI above zero and candidate ECE no worse than matched baseline",
        "baseline": baseline["aggregate"],
        "candidate": candidate["aggregate"],
        "paired_comparison": comparison,
        "elapsed_seconds": time.perf_counter() - started,
    }
    candidate_checkpoint = candidate["checkpoint"]
    candidate_checkpoint["selection_decision"] = decision["decision"]
    torch.save(candidate_checkpoint, args.candidate_checkpoint)
    write_csv(args.output_dir / "baseline_per_scene.csv", baseline["per_scene"])
    write_csv(args.output_dir / "candidate_per_scene.csv", candidate["per_scene"])
    write_csv(args.output_dir / "training_history.csv", baseline["history"] + candidate["history"])
    (args.output_dir / "detector_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.output_dir / "detector_improvement_report.md").write_text(build_report(decision), encoding="utf-8")
    print(json.dumps(decision, indent=2))


def train_recipe(name, train_rows, validation_rows, initialization, args, device):
    set_seed(args.seed)
    model = BurnScarUtilityNet(args.base_channels).to(device)
    model.load_state_dict(deepcopy(initialization))
    train_loader = make_loader(
        train_rows,
        args,
        shuffle=True,
        augmentation="basic" if name == "matched_baseline" else "strong",
        seed=args.seed,
    )
    validation_loader = make_loader(validation_rows, args, shuffle=False, augmentation="none", seed=args.seed)
    positive_weight = estimate_positive_weight(train_loader)
    baseline_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dice = -1.0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, name, baseline_bce)
        logits, targets, _, validation_loss = collect_logits(model, validation_loader, device, name, baseline_bce)
        probabilities = probabilities_from_logits(logits)
        threshold, metrics = select_threshold(probabilities, targets)
        scheduler.step(metrics["dice"])
        record = {
            "recipe": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_dice": metrics["dice"],
            "validation_iou": metrics["iou"],
            "threshold": threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["dice"] > best_dice + 1e-4:
            best_dice = metrics["dice"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    logits, targets, sample_ids, _ = collect_logits(model, validation_loader, device, name, baseline_bce)
    temperature, calibration_bce = select_temperature(logits, targets)
    probabilities = probabilities_from_logits(logits, temperature)
    threshold, aggregate = select_threshold(probabilities, targets)
    aggregate.update(
        {
            "recipe": name,
            "temperature": temperature,
            "threshold": threshold,
            "ece": expected_calibration_error(probabilities, targets),
            "brier": float(np.mean((probabilities - targets) ** 2)),
            "calibration_bce": calibration_bce,
            "train_scenes": len(train_rows),
            "validation_scenes": len(validation_rows),
            "train_groups": len({row["geographic_group"] for row in train_rows}),
            "validation_groups": len({row["geographic_group"] for row in validation_rows}),
        }
    )
    per_scene = per_scene_metrics(probabilities, targets, sample_ids, threshold, name)
    checkpoint = {
        "model_type": "burn_scar_utility_unet",
        "model": best_state,
        "config": {
            "base_channels": args.base_channels,
            "input_size": args.image_size,
            "threshold": threshold,
            "temperature": temperature,
        },
        "training": {
            "recipe": name,
            "seed": args.seed,
            "source_partition": "selector_train only",
            "internal_validation_partition": "selector_validation",
            "official_validation_used": False,
            "test_split_used": False,
            "train_scenes": len(train_rows),
            "validation_scenes": len(validation_rows),
            "best_validation_dice": aggregate["dice"],
            "validation_ece": aggregate["ece"],
        },
    }
    return {"aggregate": aggregate, "per_scene": per_scene, "checkpoint": checkpoint, "history": history}


def train_epoch(model, loader, optimizer, scaler, device, recipe, baseline_bce):
    model.train()
    losses = []
    for images, targets, _, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            if recipe == "matched_baseline":
                loss = baseline_bce(logits, targets) + dice_loss(logits, targets)
            else:
                loss = focal_tversky_hard_negative_loss(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def collect_logits(model, loader, device, recipe, baseline_bce):
    model.eval()
    logits_out, targets_out, sample_ids, losses = [], [], [], []
    for images, targets, ids, _ in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        if recipe == "matched_baseline":
            loss = baseline_bce(logits, targets) + dice_loss(logits, targets)
        else:
            loss = focal_tversky_hard_negative_loss(logits, targets)
        logits_out.append(logits.float().cpu().numpy())
        targets_out.append(targets.float().cpu().numpy())
        sample_ids.extend(ids)
        losses.append(float(loss.cpu()))
    return np.concatenate(logits_out), np.concatenate(targets_out), sample_ids, float(np.mean(losses))


def select_threshold(probabilities, targets):
    candidates = np.linspace(0.05, 0.95, 37)
    scored = [(float(threshold), aggregate_metrics(probabilities, targets, float(threshold))) for threshold in candidates]
    return max(scored, key=lambda item: item[1]["dice"])


def aggregate_metrics(probabilities, targets, threshold):
    predicted = probabilities >= threshold
    truth = targets >= 0.5
    tp = float(np.logical_and(predicted, truth).sum())
    fp = float(np.logical_and(predicted, ~truth).sum())
    fn = float(np.logical_and(~predicted, truth).sum())
    tn = float(np.logical_and(~predicted, ~truth).sum())
    epsilon = 1e-9
    return {
        "dice": 2.0 * tp / (2.0 * tp + fp + fn + epsilon),
        "iou": tp / (tp + fp + fn + epsilon),
        "precision": tp / (tp + fp + epsilon),
        "recall": tp / (tp + fn + epsilon),
        "specificity": tn / (tn + fp + epsilon),
        "balanced_accuracy": 0.5 * (tp / (tp + fn + epsilon) + tn / (tn + fp + epsilon)),
    }


def per_scene_metrics(probabilities, targets, sample_ids, threshold, recipe):
    rows = []
    for probability, truth, sample_id in zip(probabilities, targets, sample_ids):
        metrics = aggregate_metrics(probability[None], truth[None], threshold)
        rows.append({"sample_id": sample_id, "recipe": recipe, **metrics})
    return rows


def paired_comparison(baseline_rows, candidate_rows, seed):
    baseline = {row["sample_id"]: row for row in baseline_rows}
    candidate = {row["sample_id"]: row for row in candidate_rows}
    ids = sorted(set(baseline) & set(candidate))
    base = np.asarray([float(baseline[sample_id]["dice"]) for sample_id in ids])
    selected = np.asarray([float(candidate[sample_id]["dice"]) for sample_id in ids])
    differences = selected - base
    validator = StatisticalValidator()
    t_result = validator.paired_t_test(base.tolist(), selected.tolist())
    w_result = validator.wilcoxon(base.tolist(), selected.tolist())
    low, high = bootstrap_mean_ci(differences, seed=seed)
    return {
        "n": len(ids),
        "baseline_scene_dice_mean": float(base.mean()),
        "candidate_scene_dice_mean": float(selected.mean()),
        "mean_paired_dice_difference": float(differences.mean()),
        "dice_difference_ci_low": low,
        "dice_difference_ci_high": high,
        "paired_t_p_value": t_result.p_value,
        "wilcoxon_p_value": w_result.p_value,
        "cohens_dz": t_result.effect_size,
    }


def spatial_augmentation(image, mask):
    if random.random() < 0.5:
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    if random.random() < 0.5:
        image, mask = np.flip(image, 0), np.flip(mask, 0)
    rotations = random.randint(0, 3)
    return np.rot90(image, rotations, (0, 1)), np.rot90(mask, rotations, (0, 1))


def radiometric_augmentation(image):
    contrast = random.uniform(0.82, 1.18)
    brightness = random.uniform(-0.08, 0.08)
    gamma = random.uniform(0.85, 1.18)
    channel_scale = np.asarray([random.uniform(0.92, 1.08) for _ in range(3)], dtype="float32")
    values = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    values = np.power(values, gamma) * channel_scale
    if random.random() < 0.35:
        values += np.random.normal(0.0, random.uniform(0.0, 0.015), values.shape).astype("float32")
    return np.clip(values, 0.0, 1.0).astype("float32")


def make_loader(rows, args, *, shuffle, augmentation, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        GeographicBurnScarDataset(rows, args.image_size, augmentation),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def estimate_positive_weight(loader):
    positive, total = 0.0, 0.0
    for _, targets, _, _ in loader:
        positive += float(targets.sum())
        total += float(targets.numel())
    prevalence = positive / max(total, 1.0)
    return float(np.clip((1.0 - prevalence) / max(prevalence, 1e-6), 1.0, 20.0))


def assert_geographic_disjoint(train_rows, validation_rows):
    train_groups = {row["geographic_group"] for row in train_rows}
    validation_groups = {row["geographic_group"] for row in validation_rows}
    overlap = train_groups & validation_groups
    if overlap:
        raise RuntimeError(f"Geographic leakage detected: {sorted(overlap)}")


def build_report(decision):
    baseline, candidate, comparison = decision["baseline"], decision["candidate"], decision["paired_comparison"]
    return "\n".join(
        [
            "# Calibrated Burn-Scar Detector Experiment",
            "",
            "Two identical U-Net models were initialized from the same weights and trained on the same 259-scene geographic training split. The matched baseline used weighted BCE plus Dice loss and basic spatial augmentation. The candidate used focal-Tversky loss, online hard-negative emphasis, stronger radiometric augmentation, and validation-only temperature calibration.",
            "",
            "## Leakage controls",
            "",
            "- Model fitting: 259 scenes from 150 geographic groups",
            "- Model selection and calibration: 59 scenes from 38 disjoint geographic groups",
            "- Official validation used: no",
            "- Final test used: no",
            "",
            "## Aggregate internal-validation results",
            "",
            "| recipe | Dice | IoU | precision | recall | balanced accuracy | ECE | Brier |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| matched baseline | {baseline['dice']:.4f} | {baseline['iou']:.4f} | {baseline['precision']:.4f} | {baseline['recall']:.4f} | {baseline['balanced_accuracy']:.4f} | {baseline['ece']:.4f} | {baseline['brier']:.4f} |",
            f"| calibrated candidate | {candidate['dice']:.4f} | {candidate['iou']:.4f} | {candidate['precision']:.4f} | {candidate['recall']:.4f} | {candidate['balanced_accuracy']:.4f} | {candidate['ece']:.4f} | {candidate['brier']:.4f} |",
            "",
            "## Paired scene-level test",
            "",
            f"Candidate minus baseline Dice: {comparison['mean_paired_dice_difference']:+.4f}, 95% CI {comparison['dice_difference_ci_low']:+.4f} to {comparison['dice_difference_ci_high']:+.4f}, paired t-test p={comparison['paired_t_p_value']:.4g}, Wilcoxon p={comparison['wilcoxon_p_value']:.4g}.",
            "",
            "## Decision",
            "",
            f"**{decision['decision'].replace('_', ' ').title()}.** Advancement requires a positive paired Dice confidence interval and calibration error no worse than the matched baseline. The existing production detector is not replaced automatically.",
        ]
    )


def build_baseline_report(result):
    metrics = result["metrics"]
    return "\n".join(
        [
            "# Retrained and Calibrated Burn-Scar Detector",
            "",
            "The winning weighted BCE-plus-Dice recipe was retrained on 259 scenes from 150 geographic groups, selected and temperature-calibrated on 59 scenes from 38 disjoint geographic groups, and saved as a separate candidate checkpoint.",
            "",
            f"- Aggregate Dice: {metrics['dice']:.4f}",
            f"- IoU: {metrics['iou']:.4f}",
            f"- Precision: {metrics['precision']:.4f}",
            f"- Recall: {metrics['recall']:.4f}",
            f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
            f"- Expected calibration error: {metrics['ece']:.4f}",
            f"- Brier score: {metrics['brier']:.4f}",
            f"- Temperature: {metrics['temperature']:.3f}",
            f"- Threshold: {metrics['threshold']:.3f}",
            "",
            "The official validation and final test partitions remain unused. This checkpoint does not replace the registered production detector until an independent matched comparison is completed.",
        ]
    )


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()
