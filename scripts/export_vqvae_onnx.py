"""scripts.export_vqvae_onnx

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.encoder_service import EncoderService


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the VQ-VAE reconstruction path to ONNX for edge experiments.")
    parser.add_argument("--checkpoint", default="checkpoints/vqvae_s16k8.pt")
    parser.add_argument("--output", default="outputs/vqvae_reconstruct.onnx")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    service = EncoderService(Path(args.checkpoint), device="cpu")
    model = service.model.eval()
    sample = torch.randn(1, 3, args.size, args.size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["image"],
        output_names=["reconstruction", "tokens", "commitment_loss"],
        opset_version=17,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
