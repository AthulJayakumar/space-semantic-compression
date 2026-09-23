"""Train and evaluate the supervised burn-scar utility detector."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
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
from semantic_ai.burn_scar_model import BurnScarUtilityNet, dice_loss  # noqa: E402


class BurnScarDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int, augment: bool = False) -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = load_rgb_image(Path(row["image_path"]))
        mask = load_grayscale_image(Path(row["mask_path"]))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype="float32") / 255.0
        mask_array = (np.asarray(mask, dtype="float32") > 0).astype("float32")
        if self.augment:
            if random.random() < 0.5:
                image_array = np.flip(image_array, axis=1)
                mask_array = np.flip(mask_array, axis=1)
            if random.random() < 0.5:
                image_array = np.flip(image_array, axis=0)
                mask_array = np.flip(mask_array, axis=0)
            rotations = random.randint(0, 3)
            image_array = np.rot90(image_array, rotations, axes=(0, 1))
            mask_array = np.rot90(mask_array, rotations, axes=(0, 1))
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_array.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_array[None, ...]))
        return image_tensor, mask_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a geography-separated burn-scar utility detector.")
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/burn_scar_utility_detector"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    rows = read_manifest(args.manifest)
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    if not all(split_rows.values()):
        raise RuntimeError("Training, validation, and test rows are all required.")

    loaders = {
        split: DataLoader(
            BurnScarDataset(items, args.image_size, augment=split == "train"),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split, items in split_rows.items()
    }
    model = BurnScarUtilityNet(args.base_channels).to(device)
    positive_weight = estimate_positive_weight(loaders["train"], device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_dice = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, bce, scaler, device)
        validation_predictions, validation_targets, validation_loss = collect_predictions(
            model, loaders["validation"], bce, device
        )
        threshold, validation_metrics = select_threshold(validation_predictions, validation_targets)
        scheduler.step(validation_metrics["dice"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "threshold": threshold,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        write_csv(args.output_dir / "training_history.csv", history)
        print(json.dumps(record), flush=True)
        if validation_metrics["dice"] > best_dice + 1e-4:
            best_dice = validation_metrics["dice"]
            best_threshold = threshold
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_type": "burn_scar_utility_unet",
                    "model": model.state_dict(),
                    "config": {
                        "base_channels": args.base_channels,
                        "input_size": args.image_size,
                        "threshold": threshold,
                    },
                    "training": {
                        "seed": args.seed,
                        "train_items": len(split_rows["train"]),
                        "validation_items": len(split_rows["validation"]),
                        "geographic_split": True,
                        "best_epoch": epoch,
                        "best_validation_dice": best_dice,
                    },
                },
                args.checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    test_predictions, test_targets, test_loss = collect_predictions(model, loaders["test"], bce, device)
    test_metrics = segmentation_metrics(test_predictions, test_targets, best_threshold)
    test_metrics.update(
        {
            "loss": test_loss,
            "threshold_selected_on_validation": best_threshold,
            "test_items": len(split_rows["test"]),
            "train_items": len(split_rows["train"]),
            "validation_items": len(split_rows["validation"]),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    write_report(args.output_dir / "detector_report.md", checkpoint, test_metrics)
    print(json.dumps(test_metrics, indent=2))


def train_epoch(model, loader, optimizer, bce, scaler, device) -> float:
    model.train()
    losses: list[float] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = bce(logits, targets) + dice_loss(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def collect_predictions(model, loader, bce, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    losses: list[float] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = bce(logits, targets) + dice_loss(logits, targets)
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        targets_out.append(targets.float().cpu().numpy())
        losses.append(float(loss.detach().cpu()))
    return np.concatenate(predictions), np.concatenate(targets_out), float(np.mean(losses))


def select_threshold(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.1, 0.9, 17)
    scored = [(float(threshold), segmentation_metrics(predictions, targets, float(threshold))) for threshold in candidates]
    return max(scored, key=lambda item: item[1]["dice"])


def segmentation_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = predictions >= threshold
    truth = targets >= 0.5
    true_positive = float(np.logical_and(predicted, truth).sum())
    false_positive = float(np.logical_and(predicted, ~truth).sum())
    false_negative = float(np.logical_and(~predicted, truth).sum())
    true_negative = float(np.logical_and(~predicted, ~truth).sum())
    epsilon = 1e-9
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    dice = 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative + epsilon)
    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)
    specificity = true_negative / (true_negative + false_positive + epsilon)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def estimate_positive_weight(loader: DataLoader, device: torch.device) -> float:
    positive = 0.0
    total = 0.0
    for _, targets in loader:
        positive += float(targets.sum())
        total += float(targets.numel())
    prevalence = positive / max(total, 1.0)
    return float(np.clip((1.0 - prevalence) / max(prevalence, 1e-6), 1.0, 20.0))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, checkpoint: dict[str, object], metrics: dict[str, object]) -> None:
    training = checkpoint["training"]
    lines = [
        "# Supervised Burn-Scar Utility Detector",
        "",
        "The detector was trained on the geography-separated HLS training split, selected using validation Dice, and evaluated once on the untouched geographic test split.",
        "",
        f"- Training tiles: {metrics['train_items']}",
        f"- Validation tiles: {metrics['validation_items']}",
        f"- Test tiles: {metrics['test_items']}",
        f"- Best epoch: {training['best_epoch']}",
        f"- Validation-selected threshold: {metrics['threshold_selected_on_validation']:.3f}",
        f"- Test Dice: {metrics['dice']:.4f}",
        f"- Test IoU: {metrics['iou']:.4f}",
        f"- Test precision: {metrics['precision']:.4f}",
        f"- Test recall: {metrics['recall']:.4f}",
        f"- Test specificity: {metrics['specificity']:.4f}",
        f"- Test balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        "",
        "This model estimates burn-scar relevance. It is not an active-fire or smoke detector and must not be described as one.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    main()
