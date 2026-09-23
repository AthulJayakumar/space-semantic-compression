"""Train only the selector from burn-scar labels after fixed-budget reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.tensor_utils import image_to_tensor  # noqa: E402
from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from evaluation.matched_rate import read_benchmark_manifest  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402
from scripts.train_sus_aligned_selector import geographic_internal_split, load_detector  # noqa: E402
from token_selection.learned_mode_selector import (  # noqa: E402
    MODE_TO_INDEX, ModeConditionedSelectorConfig, ModeConditionedTokenScorer,
    local_token_entropy_numpy,
)
from token_selection.task_aligned import decode_gated_tokens, straight_through_topk, task_aligned_loss  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument("--vqvae-checkpoint", type=Path, default=Path("models/checkpoints/vqvae_s16k8_pruned_semantic_finetuned.pt"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/token_task_aligned_hls.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/task_aligned_selector"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--keep-ratio", type=float, default=0.40)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("keep-ratio must be in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    service = build_service(args.vqvae_checkpoint, args.device, args.output_dir / "artifacts", args.detector_checkpoint)
    device = service.encoder_service.device
    vqvae = service.encoder_service.model.eval().requires_grad_(False)
    detector, detector_config = load_detector(args.detector_checkpoint, device)
    all_items = read_benchmark_manifest(args.manifest, split="train")
    train_items, validation_items = geographic_internal_split(all_items, 0.20, args.seed)
    if args.max_train:
        train_items = train_items[: args.max_train]
    if args.max_val:
        validation_items = validation_items[: args.max_val]
    train_samples = build_samples(train_items, service, args.crop_size)
    val_samples = build_samples(validation_items, service, args.crop_size)
    config = ModeConditionedSelectorConfig(codebook_size=int(service.encoder_service.config["codebook_size"]))
    selector = ModeConditionedTokenScorer(config).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(selector, vqvae, detector, train_samples, device, args, optimizer, epoch)
        validation = run_epoch(selector, vqvae, detector, val_samples, device, args, None, epoch)
        record = {"epoch": epoch, "train_loss": training, "internal_validation_loss": validation}
        history.append(record)
        write_csv(args.output_dir / "training_history.csv", history)
        print(f"epoch={epoch} train={training:.5f} internal_val={validation:.5f}", flush=True)
        if validation < best_loss - 1e-5:
            best_loss = validation
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in selector.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No selector checkpoint was produced")
    checkpoint = {
        "model_type": "task_aligned_token_selector",
        "model_state": best_state,
        "config": config.__dict__,
        "modes": MODE_TO_INDEX,
        "training": {
            "objective": "fixed-budget post-reconstruction burn-scar BCE + Dice + small RGB L1",
            "original_manifest_splits_used": ["train"],
            "official_validation_used": False,
            "test_split_used": False,
            "train_images": len(train_samples),
            "internal_validation_images": len(val_samples),
            "train_geographic_groups": len({item.geographic_group for item in train_items}),
            "internal_validation_geographic_groups": len({item.geographic_group for item in validation_items}),
            "keep_ratio": args.keep_ratio,
            "temperature": args.temperature,
            "reconstruction_weight": args.reconstruction_weight,
            "crop_size": args.crop_size,
            "detector_config": detector_config,
            "best_epoch": best_epoch,
            "best_internal_validation_loss": best_loss,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, args.output)
    (args.output_dir / "training_metadata.json").write_text(json.dumps(checkpoint["training"], indent=2), encoding="utf-8")
    print(json.dumps(checkpoint["training"], indent=2))


def build_samples(items, service, crop_size: int) -> list[dict[str, torch.Tensor]]:
    samples = []
    for index, item in enumerate(items, 1):
        image = load_rgb_image(item.image_path)
        mask = load_grayscale_image(item.mask_path)
        width, height = image.size
        side = min(crop_size, width, height)
        left, top = (width - side) // 2, (height - side) // 2
        box = (left, top, left + side, top + side)
        image = image.crop(box).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
        mask = mask.crop(box).resize((crop_size, crop_size), Image.Resampling.NEAREST)
        image_tensor = image_to_tensor(image, service.encoder_service.device, service.encoder_service.stride)
        with torch.inference_mode():
            tokens = service.encoder_service.encode(image_tensor)
        token_shape = tuple(tokens.shape[-2:])
        before = service._detect_mission_utility(image, token_shape, "wildfire_detection")
        semantic = service.semantic_service.analyze(image, token_shape)
        utility = np.maximum(before.utility_map, semantic.importance_map).astype("float32")
        entropy = local_token_entropy_numpy(tokens, token_shape)
        detail = service.semantic_service.detail_map(image, token_shape)
        mask_array = (np.asarray(mask, dtype="float32") > 0).astype("float32")
        samples.append({
            "tokens": tokens.cpu().long(),
            "image": image_tensor.cpu(),
            "mask": torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0),
            "utility": torch.from_numpy(utility).unsqueeze(0),
            "entropy": torch.from_numpy(entropy).unsqueeze(0),
            "detail": torch.from_numpy(detail).unsqueeze(0),
        })
        print(f"cached {index}/{len(items)}: {item.sample_id}", flush=True)
    return samples


def run_epoch(selector, vqvae, detector, samples, device, args, optimizer, epoch: int) -> float:
    training = optimizer is not None
    selector.train(training)
    order = np.arange(len(samples))
    if training:
        np.random.default_rng(args.seed + epoch).shuffle(order)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index in order:
            sample = samples[int(index)]
            tokens = sample["tokens"].to(device)
            image = sample["image"].to(device)
            mask = sample["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = selector(
                tokens,
                sample["utility"].to(device),
                sample["entropy"].to(device),
                sample["detail"].to(device),
                "mission_utility",
            )
            gate = straight_through_topk(logits, args.keep_ratio, args.temperature)
            reconstruction = decode_gated_tokens(vqvae, tokens, gate)
            loss, _ = task_aligned_loss(detector, reconstruction, image, mask, args.reconstruction_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 2.0)
                optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
