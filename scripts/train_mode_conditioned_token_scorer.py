"""Train the optional mode-conditioned token-priority model.

The training target is teacher distillation from the validated selectors:
`mission_utility` and `reconstruction_balanced`. This is intentionally a
conservative first step toward a learned selector because it improves the model
component without changing the VQ-VAE checkpoint or production defaults.
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
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner


TEACHERS = {
    "mission_utility": TokenSelectionWeights.mission_utility(),
    "reconstruction_balanced": TokenSelectionWeights.reconstruction_balanced(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mode-conditioned token-priority scorer.")
    parser.add_argument("--dataset-dir", default="datasets/sentinel2_500_patch/images")
    parser.add_argument("--output", default="models/checkpoints/mode_conditioned_token_scorer.pt")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--token-embedding-dim", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    service = get_compression_service()
    device = resolve_training_device(args.device)
    codebook_size = int(service.encoder_service.config.get("codebook_size", 8192))
    config = ModeConditionedSelectorConfig(
        codebook_size=codebook_size,
        token_embedding_dim=args.token_embedding_dim,
        hidden_channels=args.hidden_channels,
    )
    model = ModeConditionedTokenScorer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    samples = build_training_samples(Path(args.dataset_dir), args.limit, service)
    if not samples:
        raise ValueError(f"No training images found in {args.dataset_dir}")

    log_rows: list[dict[str, object]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        for sample in samples:
            for mode_name, target_scores in sample["targets"].items():
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    sample["tokens"].to(device),
                    sample["utility"].to(device),
                    sample["entropy"].to(device),
                    sample["detail"].to(device),
                    mode_name,
                )
                target = target_scores.unsqueeze(0).to(device)
                loss = F.mse_loss(torch.sigmoid(logits), target)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "mean_loss": float(np.mean(epoch_losses)), "samples": len(samples)}
        log_rows.append(row)
        print(f"epoch={epoch} mean_loss={row['mean_loss']:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "modes": MODE_TO_INDEX,
            "teacher_weights": {name: weights.__dict__ for name, weights in TEACHERS.items()},
            "training": {"epochs": args.epochs, "learning_rate": args.learning_rate, "samples": len(samples)},
        },
        output_path,
    )
    write_training_log(output_path.with_suffix(".csv"), log_rows)
    write_report(output_path.with_suffix(".md"), output_path, log_rows, args)
    print(output_path)


def build_training_samples(dataset_dir: Path, limit: int | None, service) -> list[dict[str, object]]:
    image_paths = discover_images(dataset_dir, limit)
    samples: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        tensor = image_to_tensor(image, service.encoder_service.device, service.encoder_service.stride)
        tokens = service.encoder_service.encode(tensor)
        token_shape = tuple(tokens.shape[-2:])
        semantic = service.semantic_service.analyze(image, token_shape)
        detector = service._detect_mission_utility(image, token_shape, "wildfire_detection")
        utility = np.maximum(semantic.importance_map, detector.utility_map).astype("float32")
        entropy = local_token_entropy_numpy(tokens, token_shape)
        detail = service.semantic_service.detail_map(image, token_shape)
        targets = {
            mode_name: torch.from_numpy(UtilityAwareTokenPruner(weights).score_tokens(tokens, utility, detail_map=detail)).float()
            for mode_name, weights in TEACHERS.items()
        }
        samples.append(
            {
                "tokens": tokens.detach().cpu().long(),
                "utility": torch.from_numpy(utility).float(),
                "entropy": torch.from_numpy(entropy).float(),
                "detail": torch.from_numpy(detail).float(),
                "targets": targets,
            }
        )
        print(f"prepared {index}/{len(image_paths)} {image_path.name}")
    return samples


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


def resolve_training_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def write_training_log(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mean_loss", "samples"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, checkpoint_path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    final_loss = rows[-1]["mean_loss"] if rows else "n/a"
    path.write_text(
        "\n".join(
            [
                "# Mode-Conditioned Token Scorer Training",
                "",
                "## Purpose",
                "This run trains the optional learned token-priority model by distilling the validated mission and reconstruction selectors.",
                "",
                "## Configuration",
                f"- Dataset directory: `{args.dataset_dir}`",
                f"- Training images: {args.limit}",
                f"- Epochs: {args.epochs}",
                f"- Learning rate: {args.learning_rate}",
                f"- Output checkpoint: `{checkpoint_path}`",
                "",
                "## Result",
                f"- Final mean distillation loss: {final_loss}",
                "",
                "## Interpretation",
                "This checkpoint is not automatically used by the API. It is a research artifact for evaluating whether a learned, mode-conditioned selector can outperform fixed hand-weighted token scoring.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
