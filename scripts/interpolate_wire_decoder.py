"""Create one predeclared decoder midpoint without altering encoder or codebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def interpolate(base_path: Path, task_path: Path, alpha: float) -> dict[str, object]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    task = torch.load(task_path, map_location="cpu", weights_only=True)
    if base["args"] != task["args"] or base["model"].keys() != task["model"].keys():
        raise ValueError("Checkpoints must have identical VQ-VAE architecture")
    mixed = {}
    for key, original in base["model"].items():
        adapted = task["model"][key]
        if original.shape != adapted.shape:
            raise ValueError(f"Parameter shape mismatch: {key}")
        if key.startswith(("post_vq.", "decoder.")):
            mixed[key] = torch.lerp(original, adapted, alpha)
        else:
            if not torch.equal(original, adapted):
                raise ValueError(f"Frozen parameter changed: {key}")
            mixed[key] = original.clone()
    return {
        "model": mixed,
        "args": base["args"],
        "fine_tuning": {
            "method": "predeclared_decoder_weight_midpoint",
            "base_checkpoint": str(base_path),
            "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "task_checkpoint": str(task_path),
            "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "task_weight": alpha,
            "encoder_and_codebook_unchanged": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = interpolate(args.base, args.task, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(json.dumps(checkpoint["fine_tuning"], indent=2))


if __name__ == "__main__":
    main()
