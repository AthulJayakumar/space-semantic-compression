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
from datasets.research_wildfire import ManifestRow, load_rgb_image, read_manifest  # noqa: E402
from src.models.vqvae import VQVAE  # noqa: E402


class ManifestImageDataset(Dataset):
    """Image-only dataset backed by a wildfire image/mask manifest."""

    def __init__(self, rows: list[ManifestRow], image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = load_rgb_image(Path(self.rows[index].image_path)).convert("RGB")
        return self.transform(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VQ-VAE on satellite wildfire imagery.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/research_wildfire/wildfire_research_manifest.csv"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("../checkpoints/vqvae_s16k8.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/vqvae_s16k8_cems_hls_finetuned.pt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--commit-weight", type=float, default=0.0)
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
    if not train_rows or not validation_rows:
        raise ValueError("Fine-tuning requires at least one training row and one validation row.")

    model, checkpoint_args = load_model(args.base_checkpoint, device)
    if args.freeze_codebook:
        freeze_codebook(model)

    train_loader = DataLoader(
        ManifestImageDataset(train_rows, args.image_size),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        ManifestImageDataset(validation_rows, args.image_size),
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
        train_metrics = run_epoch(model, train_loader, device, args.commit_weight, optimizer, train=True, freeze_codebook=args.freeze_codebook)
        validation_metrics = run_epoch(model, validation_loader, device, args.commit_weight, optimizer, train=False, freeze_codebook=args.freeze_codebook)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "validation_reconstruction_loss": validation_metrics["reconstruction_loss"],
            "train_psnr": train_metrics["psnr"],
            "validation_psnr": validation_metrics["psnr"],
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
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
                "validation_fraction": args.validation_fraction,
                "seed": args.seed,
                "train_samples": len(train_rows),
                "validation_samples": len(validation_rows),
                "freeze_codebook": args.freeze_codebook,
                "best_validation_loss": best_validation_loss,
            },
        },
        args.output,
    )
    write_csv(args.output.with_suffix(".csv"), log_rows)
    write_report(args.output.with_suffix(".md"), args, log_rows, len(train_rows), len(validation_rows))
    print(args.output)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[VQVAE, dict[str, object]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = dict(checkpoint.get("args", {}))
    model_args = {
        "codebook_size": int(checkpoint_args.get("codes", 8192)),
        "code_dim": int(checkpoint_args.get("dim", 256)),
        "stride": int(checkpoint_args.get("stride", 16)),
    }
    model = VQVAE(**model_args).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint_args


def freeze_codebook(model: VQVAE) -> None:
    for parameter in model.vq.parameters():
        parameter.requires_grad = False


def run_epoch(
    model: VQVAE,
    loader: DataLoader,
    device: torch.device,
    commit_weight: float,
    optimizer: torch.optim.Optimizer,
    train: bool,
    freeze_codebook: bool,
) -> dict[str, float]:
    model.train(train)
    if train and freeze_codebook:
        model.vq.eval()
    losses: list[float] = []
    reconstruction_losses: list[float] = []
    psnr_values: list[float] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, _, commit = model(batch)
            l1 = F.l1_loss(reconstruction, batch)
            mse = F.mse_loss(reconstruction, batch)
            reconstruction_loss = l1 + 0.25 * mse
            loss = reconstruction_loss + commit_weight * commit
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
            psnr_values.append(psnr_from_normalized_mse(float(mse.detach().cpu())))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "reconstruction_loss": float(np.mean(reconstruction_losses)) if reconstruction_losses else 0.0,
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
                "",
                "## Result",
                f"- Final validation loss: {final_row.get('validation_loss', 'n/a')}",
                f"- Final validation reconstruction loss: {final_row.get('validation_reconstruction_loss', 'n/a')}",
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
