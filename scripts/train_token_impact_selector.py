"""Train a token selector from decoder replacement-impact supervision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.tensor_utils import image_to_tensor  # noqa: E402
from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from evaluation.matched_rate import read_benchmark_manifest  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402
from token_selection.impact_target import decoder_replacement_impact  # noqa: E402
from token_selection.learned_mode_selector import (  # noqa: E402
    MODE_TO_INDEX,
    ModeConditionedSelectorConfig,
    ModeConditionedTokenScorer,
    local_token_entropy_numpy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decoder-impact-aware semantic token selector.")
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument(
        "--vqvae-checkpoint",
        type=Path,
        default=Path("models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/checkpoints/token_impact_selector_hls.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/token_impact_selector"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--token-embedding-dim", type=int, default=16)
    parser.add_argument("--mission-weight", type=float, default=0.8)
    parser.add_argument("--context-pixels", type=int, default=16)
    parser.add_argument("--integration-steps", type=int, default=4)
    parser.add_argument("--rank-loss-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    service = build_service(args.vqvae_checkpoint, args.device, args.output_dir / "artifacts")
    device = service.encoder_service.device
    cache_path = args.output_dir / "impact_training_samples.pt"
    if args.reuse_cache and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_samples = cached["train_samples"]
        validation_samples = cached["validation_samples"]
    else:
        train_items = read_benchmark_manifest(args.manifest, split="train", limit=args.train_limit)
        validation_items = read_benchmark_manifest(args.manifest, split="validation", limit=args.validation_limit)
        train_samples = build_samples(train_items, service, args, "train")
        validation_samples = build_samples(validation_items, service, args, "validation")
        torch.save(
            {
                "version": 1,
                "vqvae_checkpoint": str(args.vqvae_checkpoint),
                "mission_weight": args.mission_weight,
                "context_pixels": args.context_pixels,
                "integration_steps": args.integration_steps,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
            },
            cache_path,
        )
    if not train_samples or not validation_samples:
        raise RuntimeError("Both geographic training and validation samples are required.")

    config = ModeConditionedSelectorConfig(
        codebook_size=int(service.encoder_service.config["codebook_size"]),
        token_embedding_dim=args.token_embedding_dim,
        hidden_channels=args.hidden_channels,
    )
    model = ModeConditionedTokenScorer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history: list[dict[str, object]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_samples, optimizer, args, device, train=True, epoch=epoch)
        validation_metrics = run_epoch(
            model, validation_samples, optimizer, args, device, train=False, epoch=epoch
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f} "
            f"validation_top20_overlap={validation_metrics['top20_overlap']:.4f}",
            flush=True,
        )
        if validation_metrics["loss"] < best_loss - 1e-6:
            best_loss = validation_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    best_epoch = int(min(history, key=lambda row: float(row["validation_loss"]))["epoch"])
    checkpoint = {
        "model_type": "decoder_impact_token_selector",
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "modes": MODE_TO_INDEX,
        "training_objective": "integrated_gradient_fallback_replacement_impact",
        "training": {
            "manifest": str(args.manifest),
            "vqvae_checkpoint": str(args.vqvae_checkpoint),
            "detector_checkpoint": "models/checkpoints/wildfire_utility_segmentation.pt",
            "train_split": "geographic train",
            "validation_split": "geographic validation",
            "test_split_used": False,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "mission_weight": args.mission_weight,
            "context_pixels": args.context_pixels,
            "integration_steps": args.integration_steps,
            "rank_loss_weight": args.rank_loss_weight,
            "temperature": args.temperature,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, args.output)
    write_csv(args.output_dir / "training_history.csv", history)
    report = build_report(args, checkpoint, history)
    (args.output_dir / "training_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(checkpoint["training"], indent=2))


def build_samples(items, service, args, split: str) -> list[dict[str, torch.Tensor | str | float]]:
    samples: list[dict[str, torch.Tensor | str | float]] = []
    model = service.encoder_service.model
    for index, item in enumerate(items, start=1):
        print(f"impact targets {split} {index}/{len(items)}: {item.sample_id}", flush=True)
        image = load_rgb_image(item.image_path)
        image_tensor = image_to_tensor(image, service.encoder_service.device, service.encoder_service.stride)
        tokens = service.encoder_service.encode(image_tensor)
        token_shape = tuple(tokens.shape[-2:])
        detector = service._detect_mission_utility(image, token_shape, "wildfire_detection")
        entropy = local_token_entropy_numpy(tokens, token_shape)
        detail = service.semantic_service.detail_map(image, token_shape)
        if item.mask_path is None:
            mission_mask = torch.zeros((1, 1, image.height, image.width), device=image_tensor.device)
        else:
            mask_image = load_grayscale_image(item.mask_path).resize(image.size)
            mask_array = (np.asarray(mask_image, dtype="float32") > 0.0).astype("float32")
            mission_mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
        impact, full_loss = decoder_replacement_impact(
            model,
            tokens,
            image_tensor,
            mission_mask,
            mission_weight=args.mission_weight,
            context_pixels=args.context_pixels,
            integration_steps=args.integration_steps,
        )
        samples.append(
            {
                "sample_id": item.sample_id,
                "tokens": tokens[0].detach().cpu().long(),
                "utility": torch.from_numpy(detector.utility_map).float(),
                "entropy": torch.from_numpy(entropy).float(),
                "detail": torch.from_numpy(detail).float(),
                "target": torch.from_numpy(impact).float(),
                "full_weighted_mse": full_loss,
            }
        )
    return samples


def run_epoch(model, samples, optimizer, args, device: torch.device, *, train: bool, epoch: int) -> dict[str, float]:
    model.train(train)
    order = np.arange(len(samples))
    if train:
        np.random.default_rng(args.seed + epoch).shuffle(order)
    totals = {"loss": [], "mse": [], "rank_kl": [], "top20_overlap": []}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for start in range(0, len(order), args.batch_size):
            batch = [samples[int(index)] for index in order[start : start + args.batch_size]]
            tokens = torch.stack([sample["tokens"] for sample in batch]).to(device)
            utility = torch.stack([sample["utility"] for sample in batch]).to(device)
            entropy = torch.stack([sample["entropy"] for sample in batch]).to(device)
            detail = torch.stack([sample["detail"] for sample in batch]).to(device)
            target = torch.stack([sample["target"] for sample in batch]).to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, utility, entropy, detail, "mission_utility")
            prediction = torch.sigmoid(logits)
            mse = F.mse_loss(prediction, target)
            target_distribution = F.softmax(target.flatten(1) / args.temperature, dim=1)
            prediction_log_distribution = F.log_softmax(logits.flatten(1) / args.temperature, dim=1)
            rank_kl = F.kl_div(prediction_log_distribution, target_distribution, reduction="batchmean")
            loss = mse + args.rank_loss_weight * rank_kl
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            overlap = topk_overlap(prediction.detach(), target.detach(), keep_ratio=0.20)
            totals["loss"].append(float(loss.detach().cpu()))
            totals["mse"].append(float(mse.detach().cpu()))
            totals["rank_kl"].append(float(rank_kl.detach().cpu()))
            totals["top20_overlap"].append(overlap)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def topk_overlap(prediction: torch.Tensor, target: torch.Tensor, keep_ratio: float) -> float:
    total = prediction.shape[-2] * prediction.shape[-1]
    count = max(1, int(round(total * keep_ratio)))
    predicted = torch.topk(prediction.flatten(1), count, dim=1).indices
    expected = torch.topk(target.flatten(1), count, dim=1).indices
    overlaps = []
    for left, right in zip(predicted, expected):
        overlaps.append(torch.isin(left, right).float().mean())
    return float(torch.stack(overlaps).mean().cpu())


def build_report(args, checkpoint: dict[str, object], history: list[dict[str, object]]) -> str:
    training = checkpoint["training"]
    best = next(row for row in history if int(row["epoch"]) == int(training["best_epoch"]))
    return "\n".join(
        [
            "# Decoder-Impact Token Selector Training",
            "",
            "## Objective",
            "",
            "The selector predicts decoder-sensitive token value using integrated gradients from the actual fallback-token latent grid to the full VQ-VAE latent representation under wildfire-weighted reconstruction loss.",
            "",
            "## Data separation",
            "",
            f"- Geographic training samples: {training['train_samples']}",
            f"- Geographic validation samples: {training['validation_samples']}",
            "- Test samples used during training or model selection: 0",
            "",
            "## Result",
            "",
            f"- Best epoch: {training['best_epoch']}",
            f"- Best validation loss: {training['best_validation_loss']:.6f}",
            f"- Validation top-20% token overlap: {float(best['validation_top20_overlap']):.4f}",
            f"- Checkpoint: `{args.output}`",
            "",
            "The checkpoint remains experimental until it is selected on matched-rate validation evidence and evaluated once on the untouched geographic test partition.",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
