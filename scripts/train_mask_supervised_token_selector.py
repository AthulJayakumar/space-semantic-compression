"""Train the AI token selector with real wildfire mask supervision.

This is the next step after teacher distillation and SUS-surrogate training. It
uses paired satellite images and wildfire/burn-scar masks to teach the selector
which VQ-VAE tokens should receive transmission priority for Earth Observation
utility.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.factory import get_compression_service  # noqa: E402
from backend.utils.tensor_utils import image_to_tensor  # noqa: E402
from datasets.research_wildfire import ManifestRow, load_grayscale_image, load_rgb_image, read_manifest  # noqa: E402
from token_selection.learned_mode_selector import (  # noqa: E402
    MODE_TO_INDEX,
    ModeConditionedSelectorConfig,
    ModeConditionedTokenScorer,
    local_token_entropy_numpy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mask-supervised wildfire token selector.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/research_wildfire/wildfire_research_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/mask_supervised_token_selector.pt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--token-embedding-dim", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--budget-weight", type=float, default=2.0)
    parser.add_argument("--focal-weight", type=float, default=1.5)
    parser.add_argument("--detector-blend", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    service = get_compression_service()
    device = resolve_training_device(args.device)
    codebook_size = int(service.encoder_service.config.get("codebook_size", 8192))
    config = ModeConditionedSelectorConfig(
        codebook_size=codebook_size,
        token_embedding_dim=args.token_embedding_dim,
        hidden_channels=args.hidden_channels,
    )
    model = ModeConditionedTokenScorer(config).to(device)
    if args.init_checkpoint is not None:
        load_initial_weights(model, args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    samples = build_training_samples(rows, service, args.detector_blend)
    if not samples:
        raise ValueError(f"No usable image/mask pairs found in {args.manifest}")
    train_samples, validation_samples = split_samples(samples, args.validation_fraction, args.seed)

    log_rows: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, optimizer, train_samples, device, args, train=True)
        validation_metrics = run_epoch(model, optimizer, validation_samples, device, args, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_mask_retention": train_metrics["mask_retention"],
            "validation_mask_retention": validation_metrics["mask_retention"],
            "train_budget_error": train_metrics["budget_error"],
            "validation_budget_error": validation_metrics["budget_error"],
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
        }
        log_rows.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} "
            "validation_mask_retention={validation_mask_retention:.4f} "
            "validation_budget_error={validation_budget_error:.6f}".format(**row)
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "modes": MODE_TO_INDEX,
            "training_objective": "mask_supervised_wildfire_token_selection",
            "training": {
                "manifest": str(args.manifest),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "samples": len(samples),
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "validation_fraction": args.validation_fraction,
                "seed": args.seed,
                "keep_ratio": args.keep_ratio,
                "budget_weight": args.budget_weight,
                "focal_weight": args.focal_weight,
                "detector_blend": args.detector_blend,
                "init_checkpoint": str(args.init_checkpoint or ""),
                "best_validation_loss": best_validation_loss,
            },
        },
        args.output,
    )
    write_training_log(args.output.with_suffix(".csv"), log_rows)
    write_report(args.output.with_suffix(".md"), args.output, log_rows, args)
    print(args.output)


def run_epoch(
    model: ModeConditionedTokenScorer,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, torch.Tensor]],
    device: torch.device,
    args: argparse.Namespace,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    totals = {"loss": [], "mask_retention": [], "budget_error": []}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for sample in samples:
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = sample_loss(model, sample, device, args.keep_ratio, args.budget_weight, args.focal_weight)
            if train:
                loss.backward()
                optimizer.step()
            totals["loss"].append(float(loss.detach().cpu()))
            totals["mask_retention"].append(metrics["mask_retention"])
            totals["budget_error"].append(metrics["budget_error"])
    return {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}


def sample_loss(
    model: ModeConditionedTokenScorer,
    sample: dict[str, torch.Tensor],
    device: torch.device,
    keep_ratio: float,
    budget_weight: float,
    focal_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(
        sample["tokens"].to(device),
        sample["utility"].to(device),
        sample["entropy"].to(device),
        sample["detail"].to(device),
        "mission_utility",
    )
    probabilities = torch.sigmoid(logits)
    target = sample["target"].unsqueeze(0).to(device)
    positive_weight = 1.0 + focal_weight * target
    bce = F.binary_cross_entropy(probabilities, target, weight=positive_weight)
    mask_retention = weighted_retention(probabilities, target)
    budget_error = probabilities.mean() - float(np.clip(keep_ratio, 0.01, 1.0))
    loss = bce + (1.0 - mask_retention) + budget_weight * budget_error.square()
    return loss, {
        "mask_retention": float(mask_retention.detach().cpu()),
        "budget_error": float(abs(budget_error.detach().cpu())),
    }


def weighted_retention(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denominator = target.sum().clamp_min(1e-6)
    return (probabilities * target).sum() / denominator


def build_training_samples(rows: list[ManifestRow], service, detector_blend: float) -> list[dict[str, torch.Tensor]]:
    samples: list[dict[str, torch.Tensor]] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row.image_path)
        mask_path = Path(row.mask_path)
        try:
            image = load_rgb_image(image_path)
            tensor = image_to_tensor(image, service.encoder_service.device, service.encoder_service.stride)
            tokens = service.encoder_service.encode(tensor)
            token_shape = tuple(tokens.shape[-2:])
            mask_utility = mask_to_token_map(mask_path, token_shape)
            semantic = service.semantic_service.analyze(image, token_shape)
            detector = service._detect_mission_utility(image, token_shape, "wildfire_detection")
            detector_utility = normalize_numpy(np.maximum(semantic.importance_map, detector.utility_map).astype("float32"))
            utility = blend_maps(mask_utility, detector_utility, detector_blend)
            entropy = local_token_entropy_numpy(tokens, token_shape)
            detail = normalize_numpy(service.semantic_service.detail_map(image, token_shape))
            samples.append(
                {
                    "tokens": tokens.detach().cpu().long(),
                    "utility": torch.from_numpy(utility).float(),
                    "entropy": torch.from_numpy(entropy).float(),
                    "detail": torch.from_numpy(detail).float(),
                    "target": torch.from_numpy(mask_utility).float(),
                }
            )
            print(f"prepared {index}/{len(rows)} {image_path.name}")
        except Exception as exc:
            print(f"skipped {image_path}: {exc}")
    return samples


def mask_to_token_map(mask_path: Path, token_shape: tuple[int, int]) -> np.ndarray:
    mask = load_grayscale_image(mask_path).resize((token_shape[1], token_shape[0]))
    values = np.asarray(mask).astype("float32") / 255.0
    return normalize_numpy(values)


def blend_maps(mask_utility: np.ndarray, detector_utility: np.ndarray, detector_blend: float) -> np.ndarray:
    blend = float(np.clip(detector_blend, 0.0, 1.0))
    return normalize_numpy((1.0 - blend) * mask_utility + blend * detector_utility)


def load_initial_weights(model: ModeConditionedTokenScorer, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"initialized from {checkpoint_path}")


def split_samples(samples: list[dict[str, torch.Tensor]], validation_fraction: float, seed: int) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    fraction = float(np.clip(validation_fraction, 0.0, 0.8))
    if len(samples) < 2 or fraction <= 0:
        return samples, []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    validation_count = max(1, int(round(len(samples) * fraction)))
    validation_indices = set(indices[:validation_count].tolist())
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in validation_indices]
    validation_samples = [sample for idx, sample in enumerate(samples) if idx in validation_indices]
    return train_samples, validation_samples


def normalize_numpy(values: np.ndarray) -> np.ndarray:
    values = values.astype("float32")
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype="float32")
    return (values - lo) / (hi - lo)


def resolve_training_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def write_training_log(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, checkpoint_path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    best_row = min(rows, key=lambda row: float(row["validation_loss"])) if rows else {}
    final_row = rows[-1] if rows else {}
    path.write_text(
        "\n".join(
            [
                "# Mask-Supervised Wildfire Token Selector Training",
                "",
                "## Purpose",
                "This run trains the learned token selector using paired satellite wildfire/burn-scar masks.",
                "",
                "## Configuration",
                f"- Manifest: `{args.manifest}`",
                f"- Training limit: {args.limit or 'all manifest rows'}",
                f"- Epochs: {args.epochs}",
                f"- Learning rate: {args.learning_rate}",
                f"- Keep ratio: {args.keep_ratio}",
                f"- Detector blend: {args.detector_blend}",
                f"- Output checkpoint: `{checkpoint_path}`",
                "",
                "## Result",
                f"- Final train loss: {final_row.get('train_loss', 'n/a')}",
                f"- Final validation loss: {final_row.get('validation_loss', 'n/a')}",
                f"- Best validation loss: {best_row.get('validation_loss', 'n/a')}",
                f"- Best validation mask retention: {best_row.get('validation_mask_retention', 'n/a')}",
                "",
                "## Interpretation",
                "This checkpoint should be evaluated with the existing held-out benchmark before replacing any default selector.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
