"""Train a SUS-aware AI token selector with a direct mission-utility objective.

Unlike the distillation trainer, this script does not ask the model to imitate
the fixed selector. It trains the same lightweight selector network with a
differentiable surrogate for Semantic Utility Score: preserve detector
confidence, wildfire utility, relevance mass, and local detail while staying
inside a target token-retention budget.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.factory import get_compression_service
from backend.utils.tensor_utils import image_to_tensor
from token_selection.learned_mode_selector import (
    MODE_TO_INDEX,
    ModeConditionedSelectorConfig,
    ModeConditionedTokenScorer,
    local_token_entropy_numpy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SUS-aware AI token selector.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/sentinel2_full_patch_256/images"))
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/sus_aware_token_selector_sentinel2_full.pt"))
    parser.add_argument("--limit", type=int, default=3224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--token-embedding-dim", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--budget-weight", type=float, default=3.0)
    parser.add_argument("--bce-weight", type=float, default=0.2)
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

    samples = build_training_samples(args.dataset_dir, args.limit, service)
    if not samples:
        raise ValueError(f"No training images found in {args.dataset_dir}")
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
            "train_soft_sus": train_metrics["soft_sus"],
            "validation_soft_sus": validation_metrics["soft_sus"],
            "train_budget_error": train_metrics["budget_error"],
            "validation_budget_error": validation_metrics["budget_error"],
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
        }
        log_rows.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} "
            "train_soft_sus={train_soft_sus:.4f} validation_soft_sus={validation_soft_sus:.4f} "
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
            "training_objective": "sus_aware_surrogate",
            "training": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "samples": len(samples),
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "validation_fraction": args.validation_fraction,
                "seed": args.seed,
                "keep_ratio": args.keep_ratio,
                "budget_weight": args.budget_weight,
                "bce_weight": args.bce_weight,
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
    if train:
        model.train()
    else:
        model.eval()

    totals = {"loss": [], "soft_sus": [], "budget_error": []}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for sample in samples:
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = sample_loss(model, sample, device, args.keep_ratio, args.budget_weight, args.bce_weight)
            if train:
                loss.backward()
                optimizer.step()
            totals["loss"].append(float(loss.detach().cpu()))
            totals["soft_sus"].append(metrics["soft_sus"])
            totals["budget_error"].append(metrics["budget_error"])

    return {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}


def sample_loss(
    model: ModeConditionedTokenScorer,
    sample: dict[str, torch.Tensor],
    device: torch.device,
    keep_ratio: float,
    budget_weight: float,
    bce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(
        sample["tokens"].to(device),
        sample["utility"].to(device),
        sample["entropy"].to(device),
        sample["detail"].to(device),
        "mission_utility",
    )
    probabilities = torch.sigmoid(logits)
    utility = sample["utility"].unsqueeze(0).to(device)
    confidence = sample["confidence"].unsqueeze(0).to(device)
    relevance = sample["relevance"].unsqueeze(0).to(device)
    detail = sample["detail"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)

    detector_retention = weighted_retention(probabilities, confidence)
    utility_retention = weighted_retention(probabilities, utility)
    relevance_retention = weighted_retention(probabilities, relevance)
    detail_retention = weighted_retention(probabilities, detail)
    soft_sus = (
        0.4 * detector_retention
        + 0.3 * utility_retention
        + 0.2 * relevance_retention
        + 0.1 * detail_retention
    )
    budget_error = probabilities.mean() - float(np.clip(keep_ratio, 0.01, 1.0))
    budget_loss = budget_error.square()
    bce_loss = F.binary_cross_entropy(probabilities, target)
    loss = (1.0 - soft_sus) + budget_weight * budget_loss + bce_weight * bce_loss
    return loss, {"soft_sus": float(soft_sus.detach().cpu()), "budget_error": float(abs(budget_error.detach().cpu()))}


def weighted_retention(probabilities: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum().clamp_min(1e-6)
    return (probabilities * weights).sum() / denominator


def build_training_samples(dataset_dir: Path, limit: int | None, service) -> list[dict[str, torch.Tensor]]:
    image_paths = discover_images(dataset_dir, limit)
    samples: list[dict[str, torch.Tensor]] = []
    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        tensor = image_to_tensor(image, service.encoder_service.device, service.encoder_service.stride)
        tokens = service.encoder_service.encode(tensor)
        token_shape = tuple(tokens.shape[-2:])
        semantic = service.semantic_service.analyze(image, token_shape)
        detector = service._detect_mission_utility(image, token_shape, "wildfire_detection")
        utility = normalize_numpy(np.maximum(semantic.importance_map, detector.utility_map).astype("float32"))
        confidence = normalize_numpy(detector.confidence_map.astype("float32"))
        relevance = normalize_numpy(detector.relevance_map.astype("float32"))
        entropy = local_token_entropy_numpy(tokens, token_shape)
        detail = normalize_numpy(service.semantic_service.detail_map(image, token_shape))
        target = make_priority_target(utility, confidence, relevance, entropy, detail)
        samples.append(
            {
                "tokens": tokens.detach().cpu().long(),
                "utility": torch.from_numpy(utility).float(),
                "confidence": torch.from_numpy(confidence).float(),
                "relevance": torch.from_numpy(relevance).float(),
                "entropy": torch.from_numpy(entropy).float(),
                "detail": torch.from_numpy(detail).float(),
                "target": torch.from_numpy(target).float(),
            }
        )
        print(f"prepared {index}/{len(image_paths)} {image_path.name}")
    return samples


def load_initial_weights(model: ModeConditionedTokenScorer, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"initialized from {checkpoint_path}")


def make_priority_target(
    utility: np.ndarray,
    confidence: np.ndarray,
    relevance: np.ndarray,
    entropy: np.ndarray,
    detail: np.ndarray,
) -> np.ndarray:
    target = 0.40 * confidence + 0.30 * utility + 0.20 * relevance + 0.05 * entropy + 0.05 * detail
    return normalize_numpy(target)


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


def discover_images(dataset_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(
        [
            *dataset_dir.rglob("*.png"),
            *dataset_dir.rglob("*.jpg"),
            *dataset_dir.rglob("*.jpeg"),
            *dataset_dir.rglob("*.webp"),
        ]
    )
    return images[:limit] if limit else images


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
                "# SUS-Aware AI Token Selector Training",
                "",
                "## Purpose",
                "This run trains a learned token selector with a direct mission-utility surrogate rather than teacher imitation.",
                "",
                "## Configuration",
                f"- Dataset directory: `{args.dataset_dir}`",
                f"- Training images: {args.limit}",
                f"- Epochs: {args.epochs}",
                f"- Learning rate: {args.learning_rate}",
                f"- Keep ratio: {args.keep_ratio}",
                f"- Budget weight: {args.budget_weight}",
                f"- BCE weight: {args.bce_weight}",
                f"- Initial checkpoint: `{args.init_checkpoint or ''}`",
                f"- Validation fraction: {args.validation_fraction}",
                f"- Output checkpoint: `{checkpoint_path}`",
                "",
                "## Result",
                f"- Final train loss: {final_row.get('train_loss', 'n/a')}",
                f"- Final validation loss: {final_row.get('validation_loss', 'n/a')}",
                f"- Best validation loss: {best_row.get('validation_loss', 'n/a')}",
                f"- Best validation soft SUS surrogate: {best_row.get('validation_soft_sus', 'n/a')}",
                "",
                "## Interpretation",
                "This checkpoint is experimental. It should only be promoted if held-out compression benchmarking improves SUS and detector retention against the fixed mission-utility selector.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
