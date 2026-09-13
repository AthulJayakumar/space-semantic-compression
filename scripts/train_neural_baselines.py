"""scripts.train_neural_baselines

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from baselines import AutoencoderCodec, LightweightCNNCodec, VariationalAutoencoderCodec
from baselines.neural_codecs import reconstruction_loss
from src.utils.data import make_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural compression baselines for fair research comparisons.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--model", choices=["autoencoder", "vae", "lightweight_cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(args.model).to(device)
    loader = make_loader(args.train, batch=args.batch, shuffle=True, workers=0, pin_memory=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.model == "vae":
                xhat, _, mu, logvar = model(x)
                loss = reconstruction_loss(x, xhat) + 1e-4 * model.kl_loss(mu, logvar)
            else:
                xhat, _ = model(x)
                loss = reconstruction_loss(x, xhat)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        print(f"epoch={epoch + 1} loss={total / max(count, 1):.6f}")
    output = Path(args.out or f"models/checkpoints/{args.model}_baseline.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_type": args.model}, output)
    print(f"wrote {output}")


def _make_model(name: str):
    if name == "autoencoder":
        return AutoencoderCodec()
    if name == "vae":
        return VariationalAutoencoderCodec()
    return LightweightCNNCodec()


if __name__ == "__main__":
    main()
