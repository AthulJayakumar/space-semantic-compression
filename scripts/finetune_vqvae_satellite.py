"""Fine-tune the VQ-VAE encoder/decoder on satellite wildfire imagery.

This script improves the reconstruction model itself while preserving checkpoint
compatibility. It starts from an existing VQ-VAE checkpoint, uses real
satellite imagery from a research manifest, and writes a new checkpoint with
the same architecture and argument schema used by the backend services.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.tensor_utils import resolve_device  # noqa: E402
from datasets.research_wildfire import ManifestRow, load_rgb_image, looks_like_mask, read_manifest  # noqa: E402
from semantic_ai.burn_scar_model import BurnScarUtilityNet  # noqa: E402
from src.models.vqvae import VQVAE  # noqa: E402


class ImagePathDataset(Dataset):
    """Image-only dataset backed by local image paths."""

    def __init__(self, image_paths: list[Path], image_size: int) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = load_rgb_image(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VQ-VAE on satellite wildfire imagery.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/research_wildfire/wildfire_research_manifest.csv"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("../checkpoints/vqvae_s16k8.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/vqvae_s16k8_cems_hls_finetuned.pt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--extra-image-dir",
        action="append",
        default=[],
        help="Additional image directory for mixed-domain fine-tuning. May be passed multiple times.",
    )
    parser.add_argument("--extra-limit", type=int, default=None, help="Optional cap per extra image directory.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--commit-weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic-detector-checkpoint",
        type=Path,
        default=None,
        help="Optional frozen burn-scar detector used for differentiable mission-map consistency.",
    )
    parser.add_argument(
        "--semantic-consistency-weight",
        type=float,
        default=0.0,
        help="Weight applied to the frozen detector probability-map consistency loss.",
    )
    parser.add_argument(
        "--semantic-positive-weight",
        type=float,
        default=2.0,
        help="Additional weight assigned to high-confidence mission-relevant pixels.",
    )
    parser.add_argument(
        "--pruned-training-weight",
        type=float,
        default=0.0,
        help="Weight for reconstruction and semantic consistency after utility-guided token pruning.",
    )
    parser.add_argument(
        "--pruned-retention",
        type=float,
        default=0.2,
        help="Fraction of utility-ranked tokens retained by the optional pruned training branch.",
    )
    parser.add_argument(
        "--wire-fallback",
        action="store_true",
        help="Use the modal retained code for pruning, so the receiver can reproduce the training input.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional frozen reference checkpoint used for original-model consistency regularization.",
    )
    parser.add_argument(
        "--teacher-consistency-weight",
        type=float,
        default=0.0,
        help="Weight for preserving the original checkpoint reconstruction behavior.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-codebook", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    train_rows, validation_rows = split_rows(rows, args.validation_fraction, args.seed)
    extra_train_paths, extra_validation_paths = split_extra_images(
        [Path(path) for path in args.extra_image_dir],
        args.extra_limit,
        args.validation_fraction,
        args.seed,
    )
    train_paths = [Path(row.image_path) for row in train_rows] + extra_train_paths
    validation_paths = [Path(row.image_path) for row in validation_rows] + extra_validation_paths
    if not train_paths or not validation_paths:
        raise ValueError("Fine-tuning requires at least one training image and one validation image.")

    model, checkpoint_args = load_model(args.base_checkpoint, device)
    teacher_model = None
    if args.teacher_consistency_weight > 0:
        teacher_path = args.teacher_checkpoint or args.base_checkpoint
        teacher_model, _ = load_model(teacher_path, device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
    semantic_detector = None
    semantic_temperature = 1.0
    if args.semantic_consistency_weight > 0 or args.pruned_training_weight > 0:
        if args.semantic_detector_checkpoint is None:
            raise ValueError("--semantic-detector-checkpoint is required when semantic consistency is enabled")
        semantic_detector, semantic_temperature = load_semantic_detector(
            args.semantic_detector_checkpoint, device
        )
    if args.freeze_codebook:
        freeze_codebook(model)

    train_loader = DataLoader(
        ImagePathDataset(train_paths, args.image_size),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        ImagePathDataset(validation_paths, args.image_size),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    log_rows: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.commit_weight,
            optimizer,
            train=True,
            freeze_codebook=args.freeze_codebook,
            teacher_model=teacher_model,
            teacher_consistency_weight=args.teacher_consistency_weight,
            semantic_detector=semantic_detector,
            semantic_temperature=semantic_temperature,
            semantic_consistency_weight=args.semantic_consistency_weight,
            semantic_positive_weight=args.semantic_positive_weight,
            pruned_training_weight=args.pruned_training_weight,
            pruned_retention=args.pruned_retention,
            wire_fallback=args.wire_fallback,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            args.commit_weight,
            optimizer,
            train=False,
            freeze_codebook=args.freeze_codebook,
            teacher_model=teacher_model,
            teacher_consistency_weight=args.teacher_consistency_weight,
            semantic_detector=semantic_detector,
            semantic_temperature=semantic_temperature,
            semantic_consistency_weight=args.semantic_consistency_weight,
            semantic_positive_weight=args.semantic_positive_weight,
            pruned_training_weight=args.pruned_training_weight,
            pruned_retention=args.pruned_retention,
            wire_fallback=args.wire_fallback,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "validation_reconstruction_loss": validation_metrics["reconstruction_loss"],
            "train_teacher_consistency_loss": train_metrics["teacher_consistency_loss"],
            "validation_teacher_consistency_loss": validation_metrics["teacher_consistency_loss"],
            "train_semantic_consistency_loss": train_metrics["semantic_consistency_loss"],
            "validation_semantic_consistency_loss": validation_metrics["semantic_consistency_loss"],
            "train_pruned_reconstruction_loss": train_metrics["pruned_reconstruction_loss"],
            "validation_pruned_reconstruction_loss": validation_metrics["pruned_reconstruction_loss"],
            "train_pruned_semantic_loss": train_metrics["pruned_semantic_loss"],
            "validation_pruned_semantic_loss": validation_metrics["pruned_semantic_loss"],
            "train_psnr": train_metrics["psnr"],
            "validation_psnr": validation_metrics["psnr"],
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "extra_train_samples": len(extra_train_paths),
            "extra_validation_samples": len(extra_validation_paths),
        }
        log_rows.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} "
            "validation_reconstruction_loss={validation_reconstruction_loss:.6f} "
            "train_psnr={train_psnr:.3f} validation_psnr={validation_psnr:.3f}".format(**row)
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": checkpoint_args,
            "fine_tuning": {
                "manifest": str(args.manifest),
                "base_checkpoint": str(args.base_checkpoint),
                "epochs": args.epochs,
                "batch": args.batch,
                "image_size": args.image_size,
                "learning_rate": args.learning_rate,
                "commit_weight": args.commit_weight,
                "teacher_checkpoint": str(args.teacher_checkpoint or ""),
                "teacher_consistency_weight": args.teacher_consistency_weight,
                "semantic_detector_checkpoint": str(args.semantic_detector_checkpoint or ""),
                "semantic_consistency_weight": args.semantic_consistency_weight,
                "semantic_positive_weight": args.semantic_positive_weight,
                "pruned_training_weight": args.pruned_training_weight,
                "pruned_retention": args.pruned_retention,
                "wire_fallback": args.wire_fallback,
                "validation_fraction": args.validation_fraction,
                "seed": args.seed,
                "train_samples": len(train_rows),
                "validation_samples": len(validation_rows),
                "extra_image_dirs": [str(path) for path in args.extra_image_dir],
                "extra_limit": args.extra_limit,
                "extra_train_samples": len(extra_train_paths),
                "extra_validation_samples": len(extra_validation_paths),
                "freeze_codebook": args.freeze_codebook,
                "best_validation_loss": best_validation_loss,
            },
        },
        args.output,
    )
    write_csv(args.output.with_suffix(".csv"), log_rows)
    write_report(args.output.with_suffix(".md"), args, log_rows, len(train_paths), len(validation_paths))
    print(args.output)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[VQVAE, dict[str, object]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_args = dict(checkpoint.get("args", {}))
    model_args = {
        "codebook_size": int(checkpoint_args.get("codes", 8192)),
        "code_dim": int(checkpoint_args.get("dim", 256)),
        "stride": int(checkpoint_args.get("stride", 16)),
    }
    model = VQVAE(**model_args).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint_args


def load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def freeze_codebook(model: VQVAE) -> None:
    for parameter in model.vq.parameters():
        parameter.requires_grad = False


def load_semantic_detector(
    checkpoint_path: Path, device: torch.device
) -> tuple[BurnScarUtilityNet, float]:
    """Load a frozen differentiable detector without changing its calibration."""

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("model_type") != "burn_scar_utility_unet":
        raise ValueError(f"Unsupported semantic detector checkpoint: {checkpoint_path}")
    config = checkpoint.get("config", {})
    detector = BurnScarUtilityNet(int(config.get("base_channels", 16))).to(device)
    detector.load_state_dict(checkpoint["model"])
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad = False
    return detector, max(float(config.get("temperature", 1.0)), 1e-6)


def semantic_consistency_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    detector: BurnScarUtilityNet,
    temperature: float,
    positive_weight: float,
) -> torch.Tensor:
    """Preserve the frozen detector response while retaining input gradients."""

    target_rgb = ((target + 1.0) * 0.5).clamp(0.0, 1.0)
    reconstruction_rgb = ((reconstruction + 1.0) * 0.5).clamp(0.0, 1.0)
    with torch.no_grad():
        target_probability = torch.sigmoid(detector(target_rgb) / temperature)
    reconstruction_probability = torch.sigmoid(detector(reconstruction_rgb) / temperature)
    weights = 1.0 + max(float(positive_weight), 0.0) * target_probability
    return (F.smooth_l1_loss(reconstruction_probability, target_probability, reduction="none") * weights).mean()


def prune_tokens_by_utility(
    tokens: torch.Tensor,
    utility_map: torch.Tensor,
    keep_ratio: float,
    wire_fallback: bool = False,
) -> torch.Tensor:
    """Replace low-utility tokens with each sample's modal fallback code."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected B x H x W tokens, received {tuple(tokens.shape)}")
    ratio = float(np.clip(keep_ratio, 0.01, 1.0))
    utility = F.interpolate(utility_map, size=tokens.shape[-2:], mode="bilinear", align_corners=False)
    flat_utility = utility[:, 0].reshape(tokens.shape[0], -1)
    flat_tokens = tokens.reshape(tokens.shape[0], -1)
    keep_count = max(1, int(round(flat_tokens.shape[1] * ratio)))
    selected = flat_utility.topk(keep_count, dim=1, largest=True, sorted=False).indices
    keep_mask = torch.zeros_like(flat_tokens, dtype=torch.bool)
    keep_mask.scatter_(1, selected, True)
    if wire_fallback:
        selected_codes = flat_tokens.gather(1, selected)
        fallback = torch.mode(selected_codes, dim=1).values.unsqueeze(1)
    else:
        fallback = torch.mode(flat_tokens, dim=1).values.unsqueeze(1)
    return torch.where(keep_mask, flat_tokens, fallback).view_as(tokens).long()


def run_epoch(
    model: VQVAE,
    loader: DataLoader,
    device: torch.device,
    commit_weight: float,
    optimizer: torch.optim.Optimizer,
    train: bool,
    freeze_codebook: bool,
    teacher_model: VQVAE | None = None,
    teacher_consistency_weight: float = 0.0,
    semantic_detector: BurnScarUtilityNet | None = None,
    semantic_temperature: float = 1.0,
    semantic_consistency_weight: float = 0.0,
    semantic_positive_weight: float = 2.0,
    pruned_training_weight: float = 0.0,
    pruned_retention: float = 0.2,
    wire_fallback: bool = False,
) -> dict[str, float]:
    model.train(train)
    if train and freeze_codebook:
        model.vq.eval()
    losses: list[float] = []
    reconstruction_losses: list[float] = []
    teacher_losses: list[float] = []
    semantic_losses: list[float] = []
    pruned_reconstruction_losses: list[float] = []
    pruned_semantic_losses: list[float] = []
    psnr_values: list[float] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, tokens, commit = model(batch)
            l1 = F.l1_loss(reconstruction, batch)
            mse = F.mse_loss(reconstruction, batch)
            reconstruction_loss = l1 + 0.25 * mse
            teacher_loss = torch.zeros((), device=device)
            if teacher_model is not None and teacher_consistency_weight > 0:
                with torch.no_grad():
                    teacher_reconstruction, _, _ = teacher_model(batch)
                teacher_l1 = F.l1_loss(reconstruction, teacher_reconstruction)
                teacher_mse = F.mse_loss(reconstruction, teacher_reconstruction)
                teacher_loss = teacher_l1 + 0.25 * teacher_mse
            semantic_loss = torch.zeros((), device=device)
            if semantic_detector is not None and semantic_consistency_weight > 0:
                semantic_loss = semantic_consistency_loss(
                    reconstruction,
                    batch,
                    semantic_detector,
                    semantic_temperature,
                    semantic_positive_weight,
                )
            pruned_reconstruction_loss = torch.zeros((), device=device)
            pruned_semantic_loss = torch.zeros((), device=device)
            if semantic_detector is not None and pruned_training_weight > 0:
                with torch.no_grad():
                    target_rgb = ((batch + 1.0) * 0.5).clamp(0.0, 1.0)
                    utility_map = torch.sigmoid(semantic_detector(target_rgb) / semantic_temperature)
                    pruned_tokens = prune_tokens_by_utility(tokens, utility_map, pruned_retention, wire_fallback)
                pruned_reconstruction = model.decode(pruned_tokens)
                pruned_l1 = F.l1_loss(pruned_reconstruction, batch)
                pruned_mse = F.mse_loss(pruned_reconstruction, batch)
                pruned_reconstruction_loss = pruned_l1 + 0.25 * pruned_mse
                pruned_semantic_loss = semantic_consistency_loss(
                    pruned_reconstruction,
                    batch,
                    semantic_detector,
                    semantic_temperature,
                    semantic_positive_weight,
                )
            loss = (
                reconstruction_loss
                + commit_weight * commit
                + teacher_consistency_weight * teacher_loss
                + semantic_consistency_weight * semantic_loss
                + pruned_training_weight
                * (pruned_reconstruction_loss + semantic_consistency_weight * pruned_semantic_loss)
            )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
            teacher_losses.append(float(teacher_loss.detach().cpu()))
            semantic_losses.append(float(semantic_loss.detach().cpu()))
            pruned_reconstruction_losses.append(float(pruned_reconstruction_loss.detach().cpu()))
            pruned_semantic_losses.append(float(pruned_semantic_loss.detach().cpu()))
            psnr_values.append(psnr_from_normalized_mse(float(mse.detach().cpu())))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "reconstruction_loss": float(np.mean(reconstruction_losses)) if reconstruction_losses else 0.0,
        "teacher_consistency_loss": float(np.mean(teacher_losses)) if teacher_losses else 0.0,
        "semantic_consistency_loss": float(np.mean(semantic_losses)) if semantic_losses else 0.0,
        "pruned_reconstruction_loss": (
            float(np.mean(pruned_reconstruction_losses)) if pruned_reconstruction_losses else 0.0
        ),
        "pruned_semantic_loss": float(np.mean(pruned_semantic_losses)) if pruned_semantic_losses else 0.0,
        "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
    }


def psnr_from_normalized_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    # Tensors are normalized to [-1, 1], so max signal difference is 2.
    return float(10.0 * math.log10(4.0 / mse))


def split_rows(rows: list[ManifestRow], validation_fraction: float, seed: int) -> tuple[list[ManifestRow], list[ManifestRow]]:
    fraction = float(np.clip(validation_fraction, 0.05, 0.8))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    validation_count = max(1, int(round(len(rows) * fraction)))
    validation_indices = set(indices[:validation_count].tolist())
    train_rows = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation_rows = [row for index, row in enumerate(rows) if index in validation_indices]
    return train_rows, validation_rows


def split_extra_images(
    image_dirs: list[Path],
    limit: int | None,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    train_paths: list[Path] = []
    validation_paths: list[Path] = []
    for index, image_dir in enumerate(image_dirs):
        paths = discover_images(image_dir, limit)
        train, validation = split_paths(paths, validation_fraction, seed + index + 1)
        train_paths.extend(train)
        validation_paths.extend(validation)
    return train_paths, validation_paths


def discover_images(image_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(
        [
            *image_dir.rglob("*.png"),
            *image_dir.rglob("*.jpg"),
            *image_dir.rglob("*.jpeg"),
            *image_dir.rglob("*.webp"),
            *image_dir.rglob("*.tif"),
            *image_dir.rglob("*.tiff"),
        ]
    )
    images = [path for path in images if not looks_like_mask(path)]
    return images[:limit] if limit else images


def split_paths(paths: list[Path], validation_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not paths:
        return [], []
    fraction = float(np.clip(validation_fraction, 0.05, 0.8))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    validation_count = max(1, int(round(len(paths) * fraction)))
    validation_indices = set(indices[:validation_count].tolist())
    train_paths = [path for index, path in enumerate(paths) if index not in validation_indices]
    validation_paths = [path for index, path in enumerate(paths) if index in validation_indices]
    return train_paths, validation_paths


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, object]], train_count: int, validation_count: int) -> None:
    best_row = min(rows, key=lambda row: float(row["validation_loss"])) if rows else {}
    final_row = rows[-1] if rows else {}
    path.write_text(
        "\n".join(
            [
                "# Satellite VQ-VAE Fine-Tuning",
                "",
                "## Purpose",
                "This run fine-tunes the VQ-VAE encoder/decoder on satellite wildfire imagery while preserving the existing architecture and checkpoint format.",
                "",
                "## Configuration",
                f"- Manifest: `{args.manifest}`",
                f"- Base checkpoint: `{args.base_checkpoint}`",
                f"- Output checkpoint: `{args.output}`",
                f"- Train samples: {train_count}",
                f"- Validation samples: {validation_count}",
                f"- Epochs: {args.epochs}",
                f"- Batch size: {args.batch}",
                f"- Image size: {args.image_size}",
                f"- Learning rate: {args.learning_rate}",
                f"- Codebook frozen: {args.freeze_codebook}",
                f"- Teacher checkpoint: `{args.teacher_checkpoint or ''}`",
                f"- Teacher consistency weight: {args.teacher_consistency_weight}",
                f"- Semantic detector checkpoint: `{args.semantic_detector_checkpoint or ''}`",
                f"- Semantic consistency weight: {args.semantic_consistency_weight}",
                f"- Semantic positive-region weight: {args.semantic_positive_weight}",
                f"- Pruned training weight: {args.pruned_training_weight}",
                f"- Pruned token retention: {args.pruned_retention}",
                "",
                "## Result",
                f"- Final validation loss: {final_row.get('validation_loss', 'n/a')}",
                f"- Final validation reconstruction loss: {final_row.get('validation_reconstruction_loss', 'n/a')}",
                f"- Final validation semantic consistency loss: {final_row.get('validation_semantic_consistency_loss', 'n/a')}",
                f"- Final validation pruned reconstruction loss: {final_row.get('validation_pruned_reconstruction_loss', 'n/a')}",
                f"- Final validation pruned semantic loss: {final_row.get('validation_pruned_semantic_loss', 'n/a')}",
                f"- Final validation PSNR: {final_row.get('validation_psnr', 'n/a')}",
                f"- Best validation loss: {best_row.get('validation_loss', 'n/a')}",
                f"- Best validation PSNR: {best_row.get('validation_psnr', 'n/a')}",
                "",
                "## Interpretation",
                "This checkpoint should be benchmarked against the original VQ-VAE before being promoted. Reconstruction loss alone is not enough; SUS, detector retention, PSNR, SSIM, and compression metrics should be compared on held-out images.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
