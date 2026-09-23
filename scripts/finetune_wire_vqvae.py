"""Fine-tune the existing VQ decoder for receiver-visible sparse payloads."""

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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.label_fidelity import binary_overlap  # noqa: E402
from evaluation.matched_rate import read_benchmark_manifest  # noqa: E402
from backend.utils.tensor_utils import image_to_tensor  # noqa: E402
from datasets.research_wildfire import load_grayscale_image, load_rgb_image  # noqa: E402
from scripts.run_matched_rate_benchmark import build_service  # noqa: E402
from scripts.train_sus_aligned_selector import geographic_internal_split, load_detector  # noqa: E402
from semantic_ai.burn_scar_model import dice_loss  # noqa: E402
from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner  # noqa: E402


def received_tokens(tokens: torch.Tensor, ranking: np.ndarray, keep_ratio: float) -> torch.Tensor:
    """Replace missing codes with the modal *selected* code, never a hidden code."""
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError("Expected a single [1,H,W] token grid")
    flat = tokens.reshape(-1)
    count = max(1, int(round(flat.numel() * keep_ratio)))
    indices = torch.as_tensor(ranking[:count], device=flat.device, dtype=torch.long)
    selected = flat[indices]
    values, frequencies = torch.unique(selected, sorted=True, return_counts=True)
    fallback = values[frequencies.argmax()]
    output = torch.full_like(flat, fallback)
    output[indices] = selected
    return output.view_as(tokens)


def prepare_samples(items, service, image_size: int):
    samples = []
    scorer = UtilityAwareTokenPruner(TokenSelectionWeights.mission_utility())
    for index, item in enumerate(items, 1):
        image = load_rgb_image(item.image_path)
        mask = load_grayscale_image(item.mask_path)
        side = min(image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        box = (left, top, left + side, top + side)
        image = image.crop(box).resize((image_size, image_size), Image.Resampling.LANCZOS)
        mask = mask.crop(box).resize((image_size, image_size), Image.Resampling.NEAREST)
        image_tensor = image_to_tensor(image, service.encoder_service.device, service.encoder_service.stride)
        with torch.inference_mode():
            tokens = service.encoder_service.encode(image_tensor)
        shape = tuple(tokens.shape[-2:])
        before = service._detect_mission_utility(image, shape, "wildfire_detection")
        semantic = service.semantic_service.analyze(image, shape)

        def normalize(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype="float32")
            span = float(values.max() - values.min())
            return (values - values.min()) / span if span > 1e-8 else np.zeros_like(values)

        utility = np.maximum(normalize(before.utility_map), normalize(semantic.importance_map))
        score = scorer.score_tokens(tokens, utility)
        target = torch.from_numpy((np.asarray(mask, dtype="uint8") > 0).astype("float32"))[None, None]
        samples.append({
            "tokens": tokens.cpu(),
            "image": image_tensor.cpu(),
            "mask": target,
            "ranking": np.argsort(-score.reshape(-1), kind="stable"),
            "positive": bool(target.any()),
        })
        print(f"cached {index}/{len(items)}: {item.sample_id}", flush=True)
    return samples


def run_epoch(model, detector, samples, optimizer, device, epoch: int, seed: int, temperature: float, threshold: float,
              keep_ratios: tuple[float, ...] = (0.3, 0.4, 0.6)):
    training = optimizer is not None
    model.eval()
    model.post_vq.train(training)
    model.decoder.train(training)
    order = np.arange(len(samples))
    if training:
        np.random.default_rng(seed + epoch).shuffle(order)
    losses: list[float] = []
    dice_all: list[float] = []
    dice_positive: list[float] = []
    ratios = keep_ratios
    with torch.set_grad_enabled(training):
        for position, index in enumerate(order):
            sample = samples[int(index)]
            original = sample["image"].to(device)
            target = sample["mask"].to(device)
            tokens = sample["tokens"].to(device)
            ratio = ratios[(position + epoch) % len(ratios)] if training else ratios[len(ratios) // 2]
            received = received_tokens(tokens, sample["ranking"], ratio)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model.decode(received)
            logits = detector(((reconstruction + 1) * 0.5).clamp(0, 1))
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=torch.tensor(5.0, device=device)
            )
            mask_loss = dice_loss(logits, target)
            reconstruction_l1 = F.l1_loss(reconstruction, original)
            full_l1 = F.l1_loss(model.decode(tokens), original)
            loss = 0.2 * (bce + mask_loss) + 0.5 * reconstruction_l1 + 0.5 * full_l1
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.post_vq.parameters()) + list(model.decoder.parameters()), 1.0
                )
                optimizer.step()
            losses.append(float(loss.detach()))
            predicted = (torch.sigmoid(logits.detach() / temperature) >= threshold).cpu().numpy()[0, 0]
            truth = target.cpu().numpy()[0, 0] > 0
            dice, _ = binary_overlap(predicted, truth)
            dice_all.append(dice)
            if sample["positive"]:
                dice_positive.append(dice)
    return {
        "loss": float(np.mean(losses)),
        "dice_all": float(np.mean(dice_all)),
        "dice_positive": float(np.mean(dice_positive)),
        "positive_items": len(dice_positive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/validated_hls_500/validated_scene_manifest.csv"))
    parser.add_argument("--source-split", default="train", help="Manifest split used for training and internal model selection.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("models/checkpoints/vqvae_s16k8_pruned_semantic_finetuned.pt"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/vqvae_wire_task_finetuned.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/vqvae_wire_task_finetuned"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.3, 0.4, 0.6])
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if not args.keep_ratios or any(not 0 < ratio <= 1 for ratio in args.keep_ratios):
        parser.error("--keep-ratios must contain fractions in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    service = build_service(args.base_checkpoint, "auto", args.output_dir / "artifacts", args.detector_checkpoint)
    device = service.encoder_service.device
    model = service.encoder_service.model
    model.eval().requires_grad_(False)
    model.post_vq.requires_grad_(True)
    model.decoder.requires_grad_(True)
    detector, config = load_detector(args.detector_checkpoint, device)
    all_items = read_benchmark_manifest(args.manifest, split=args.source_split)
    if not all_items:
        raise ValueError(f"No images found in source split {args.source_split!r}")
    train_items, val_items = geographic_internal_split(all_items, 0.2, args.seed)
    if args.max_train:
        train_items = train_items[:args.max_train]
    if args.max_val:
        val_items = val_items[:args.max_val]
    train_samples = prepare_samples(train_items, service, args.image_size)
    val_samples = prepare_samples(val_items, service, args.image_size)
    optimizer = torch.optim.AdamW(
        list(model.post_vq.parameters()) + list(model.decoder.parameters()), lr=args.learning_rate
    )
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, detector, train_samples, optimizer, device, epoch, args.seed,
                          float(config.get("temperature", 1.0)), float(config.get("threshold", 0.5)),
                          tuple(args.keep_ratios))
        val = run_epoch(model, detector, val_samples, None, device, epoch, args.seed,
                        float(config.get("temperature", 1.0)), float(config.get("threshold", 0.5)),
                        tuple(args.keep_ratios))
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train.items()},
               **{f"internal_val_{key}": value for key, value in val.items()}}
        history.append(row)
        with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
        if val["dice_positive"] > best_score + 1e-5:
            best_score = val["dice_positive"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    metadata = {
        "base_checkpoint": str(args.base_checkpoint),
        "base_sha256": hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest(),
        "training_partition": args.source_split,
        "official_validation_used": False,
        "train_images": len(train_samples),
        "internal_validation_images": len(val_samples),
        "train_geographic_groups": len({item.geographic_group for item in train_items}),
        "internal_validation_geographic_groups": len({item.geographic_group for item in val_items}),
        "frozen": "encoder, pre-vq, codebook, burn-scar detector",
        "trained": "post-vq projection and decoder",
        "wire_fallback": "modal among selected transmitted codes",
        "keep_ratios": args.keep_ratios,
        "selection_ratio": args.keep_ratios[len(args.keep_ratios) // 2],
        "selection_metric": "positive-scene burn-scar Dice on internal geographic validation",
        "best_epoch": best_epoch,
        "best_internal_validation_dice_positive": best_score,
        "seed": args.seed,
    }
    torch.save({"model": best_state, "args": base["args"], "fine_tuning": metadata}, args.output)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
