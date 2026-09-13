"""src.utils.metrics

Plain-English purpose: Original VQ-VAE model and compatibility utilities.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from lpips import LPIPS
import torch

_lpips = None
def lpips_init():
    global _lpips
    if _lpips is None:
        _lpips = LPIPS(net="alex")
    return _lpips

@torch.no_grad()
def lpips_val(x, xhat):
    net = lpips_init().to(x.device)
    return net(x, xhat).mean().item()

def l1_loss(x, xhat):
    return (x - xhat).abs().mean()
