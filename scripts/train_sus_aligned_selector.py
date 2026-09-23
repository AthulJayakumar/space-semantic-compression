"""Train a SUS-aligned token selector using only geographic training regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from semantic_ai.burn_scar_model import BurnScarUtilityNet  # noqa: E402
from token_selection.learned_mode_selector import (  # noqa: E402
    MODE_TO_INDEX,
    ModeConditionedSelectorConfig,
    ModeConditionedTokenScorer,
    local_token_entropy_numpy,
)
from token_selection.sus_target import sus_aligned_replacement_impact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a detector- and SUS-aligned token selector.")
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument(
        "--vqvae-checkpoint",
        type=Path,
        default=Path("models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt"),
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path("models/checkpoints/wildfire_utility_segmentation.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/token_sus_selector_hls.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/token_sus_selector"))
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--token-embedding-dim", type=int, default=16)
    parser.add_argument("--internal-validation-fraction", type=float, default=0.20)
    parser.add_argument("--integration-steps", type=int, default=4)
    parser.add_argument("--reconstruction-weight", type=float, default=0.15)
    parser.add_argument("--rank-loss-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    service = build_service(args.vqvae_checkpoint, args.device, args.output_dir / "artifacts")
    device = service.encoder_service.device
    detector, detector_config = load_detector(args.detector_checkpoint, device)
    all_training_items = read_benchmark_manifest(args.manifest, split="train", limit=args.limit)
    selector_train_items, selector_validation_items = geographic_internal_split(
        all_training_items, args.internal_validation_fraction, args.seed
    )
    write_internal_manifest(args.output_dir / "internal_geographic_split.csv", selector_train_items, selector_validation_items)

    cache_path = args.output_dir / "sus_training_samples.pt"
    if args.reuse_cache and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_samples = cached["train_samples"]
        validation_samples = cached["validation_samples"]
    else:
        train_samples = build_samples(selector_train_items, service, detector, detector_config, args, "train")
        validation_samples = build_samples(
            selector_validation_items, service, detector, detector_config, args, "internal_validation"
        )
        torch.save(
            {
                "version": 1,
                "objective": "differentiable_sus_aligned_integrated_gradients",
                "original_manifest_splits_used": ["train"],
                "test_or_official_validation_used": False,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
            },
            cache_path,
        )
    if not train_samples or not validation_samples:
        raise RuntimeError("The internal geographic split must contain training and validation samples.")

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
        validation_metrics = run_epoch(model, validation_samples, optimizer, args, device, train=False, epoch=epoch)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        write_csv(args.output_dir / "training_history.csv", history)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"internal_validation_loss={validation_metrics['loss']:.6f} "
            f"top20_overlap={validation_metrics['top20_overlap']:.4f}",
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
    best_epoch = int(min(history, key=lambda row: float(row["validation_loss"]))["epoch"])
    model.load_state_dict(best_state)
    checkpoint = {
        "model_type": "sus_aligned_token_selector",
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "modes": MODE_TO_INDEX,
        "training_objective": "differentiable_sus_proxy_integrated_gradient_impact",
        "training": {
            "manifest": str(args.manifest),
            "original_manifest_splits_used": ["train"],
            "official_validation_used": False,
            "test_split_used": False,
            "internal_split_unit": "geographic_group",
            "internal_train_samples": len(train_samples),
            "internal_validation_samples": len(validation_samples),
            "internal_train_groups": len({item.geographic_group for item in selector_train_items}),
            "internal_validation_groups": len({item.geographic_group for item in selector_validation_items}),
            "sus_weights": [0.4, 0.3, 0.2, 0.1],
            "reconstruction_weight": args.reconstruction_weight,
            "integration_steps": args.integration_steps,
            "rank_loss_weight": args.rank_loss_weight,
            "best_epoch": best_epoch,
            "best_internal_validation_loss": best_loss,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, args.output)
    (args.output_dir / "training_report.md").write_text(build_report(args, checkpoint, history), encoding="utf-8")
    print(json.dumps(checkpoint["training"], indent=2))


def geographic_internal_split(items, validation_fraction: float, seed: int):
    """Create deterministic, non-overlapping selector splits by geographic group."""
    groups: dict[str, list[object]] = {}
    for item in items:
        groups.setdefault(item.geographic_group, []).append(item)
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    validation_group_count = max(1, int(round(len(ordered) * float(validation_fraction))))
    validation_groups = set(ordered[:validation_group_count])
    train = [item for item in items if item.geographic_group not in validation_groups]
    validation = [item for item in items if item.geographic_group in validation_groups]
    if {item.geographic_group for item in train} & {item.geographic_group for item in validation}:
        raise RuntimeError("Internal geographic split leakage detected.")
    return train, validation


def load_detector(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "burn_scar_utility_unet":
        raise ValueError(f"Unsupported detector checkpoint: {path}")
    config = dict(checkpoint["config"])
    model = BurnScarUtilityNet(int(config.get("base_channels", 16)))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval().requires_grad_(False)
    return model, config


def build_samples(items, service, detector, detector_config, args, split: str):
    samples = []
    model = service.encoder_service.model
    for index, item in enumerate(items, start=1):
        print(f"SUS targets {split} {index}/{len(items)}: {item.sample_id}", flush=True)
        image = load_rgb_image(item.image_path)
        image_tensor = image_to_tensor(image, service.encoder_service.device, service.encoder_service.stride)
        with torch.no_grad():
            tokens = service.encoder_service.encode(image_tensor)
        token_shape = tuple(tokens.shape[-2:])
        detector_output = service._detect_mission_utility(image, token_shape, "wildfire_detection")
        entropy = local_token_entropy_numpy(tokens, token_shape)
        detail = service.semantic_service.detail_map(image, token_shape)
        mask_image = load_grayscale_image(item.mask_path).resize(image.size)
        mask_array = (np.asarray(mask_image, dtype="float32") > 0.0).astype("float32")
        mission_mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
        target, components = sus_aligned_replacement_impact(
            model,
            detector,
            tokens,
            image_tensor,
            mission_mask,
            detector_threshold=float(detector_config.get("threshold", 0.5)),
            detector_input_size=int(detector_config.get("input_size", 256)),
            integration_steps=args.integration_steps,
            reconstruction_weight=args.reconstruction_weight,
        )
        samples.append(
            {
                "sample_id": item.sample_id,
                "geographic_group": item.geographic_group,
                "tokens": tokens[0].detach().cpu().long(),
                "utility": torch.from_numpy(detector_output.utility_map).float(),
                "entropy": torch.from_numpy(entropy).float(),
                "detail": torch.from_numpy(detail).float(),
                "target": torch.from_numpy(target).float(),
                **components,
            }
        )
    return samples


def run_epoch(model, samples, optimizer, args, device: torch.device, *, train: bool, epoch: int):
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
            totals["loss"].append(float(loss.detach().cpu()))
            totals["mse"].append(float(mse.detach().cpu()))
            totals["rank_kl"].append(float(rank_kl.detach().cpu()))
            totals["top20_overlap"].append(topk_overlap(prediction.detach(), target.detach(), 0.20))
    return {key: float(np.mean(values)) for key, values in totals.items()}


def topk_overlap(prediction: torch.Tensor, target: torch.Tensor, keep_ratio: float) -> float:
    count = max(1, int(round(prediction.shape[-2] * prediction.shape[-1] * keep_ratio)))
    predicted = torch.topk(prediction.flatten(1), count, dim=1).indices
    expected = torch.topk(target.flatten(1), count, dim=1).indices
    return float(torch.stack([torch.isin(left, right).float().mean() for left, right in zip(predicted, expected)]).mean().cpu())


def write_internal_manifest(path: Path, train_items, validation_items) -> None:
    rows = []
    for split, items in (("selector_train", train_items), ("selector_validation", validation_items)):
        for item in items:
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "dataset": item.dataset,
                    "image_path": item.image_path,
                    "mask_path": item.mask_path,
                    "split": split,
                    "source_split": "train",
                    "geographic_group": item.geographic_group,
                    "event_group": item.event_group,
                    "label_source": item.label_source,
                    "image_unit": item.image_unit,
                    "cloud_fraction": item.cloud_fraction,
                    "smoke_label_status": item.smoke_label_status,
                }
            )
    write_csv(path, rows)


def build_report(args, checkpoint, history) -> str:
    training = checkpoint["training"]
    best = next(row for row in history if int(row["epoch"]) == int(training["best_epoch"]))
    return "\n".join(
        [
            "# SUS-Aligned Token Selector Training",
            "",
            "The selector was trained using integrated-gradient token targets from a frozen wildfire detector and VQ-VAE decoder. The differentiable objective mirrors the fixed SUS component weights; formal evaluation still uses the unchanged SUS implementation.",
            "",
            "## Leakage controls",
            "",
            "- Source data used: original geographic training partition only",
            f"- Internal training: {training['internal_train_samples']} scenes across {training['internal_train_groups']} geographic groups",
            f"- Internal validation: {training['internal_validation_samples']} scenes across {training['internal_validation_groups']} geographic groups",
            "- Original 98-scene validation partition used: no",
            "- Original 84-scene test partition used: no",
            "",
            "## Training result",
            "",
            f"- Best epoch: {training['best_epoch']}",
            f"- Best internal validation loss: {training['best_internal_validation_loss']:.6f}",
            f"- Internal validation top-20% overlap: {float(best['validation_top20_overlap']):.4f}",
            f"- Checkpoint: `{args.output}`",
            "",
            "Advancement requires matched-rate improvement over both fixed utility and entropy selection on the internal geographic validation split before any official validation run.",
        ]
    )


def write_csv(path: Path, rows) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
