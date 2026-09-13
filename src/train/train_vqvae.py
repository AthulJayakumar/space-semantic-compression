"""src.train.train_vqvae

Plain-English purpose: Original VQ-VAE model and compatibility utilities.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

import argparse, os
import torch
from torchvision.utils import save_image
from src.models.vqvae import VQVAE
from src.utils.data import make_loader
from src.utils.metrics import l1_loss, lpips_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--codes", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="checkpoints/vqvae_s16k8.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = make_loader(args.train, batch=args.batch, shuffle=True,  workers=0, pin_memory=False)
    val_loader   = make_loader(args.val,   batch=32,         shuffle=False, workers=0, pin_memory=False)

    model = VQVAE(codebook_size=args.codes, code_dim=args.dim, stride=args.stride).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=1e-2)

    best_lpips = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            xhat, _, commit = model(x)
            loss = l1_loss(x, xhat) + 1.0 * commit

            # optional guard
            if torch.isnan(loss):
                print("NaN detected; skipping batch"); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            lp = 0.0; n = 0
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                xhat, _, _ = model(x)
                lp += lpips_val(x, xhat); n += 1
            lp = lp / max(n, 1)

        print(f"Epoch {epoch}: val LPIPS={lp:.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        if lp < best_lpips:
            best_lpips = lp
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
            os.makedirs("out", exist_ok=True)
            grid = torch.cat([x[:8], xhat[:8]], dim=0)
            grid = (grid*0.5+0.5).clamp(0,1)
            save_image(grid, f"out/preview_epoch{epoch:03d}.png", nrow=8)

if __name__ == "__main__":
    main()
