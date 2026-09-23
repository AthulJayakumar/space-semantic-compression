"""src.models.vqvae

Plain-English purpose: Original VQ-VAE model and compatibility utilities.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --------- helpers ---------
def conv3(in_ch, out_ch, s=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(conv3(ch, ch), conv3(ch, ch))

    def forward(self, x):
        return x + self.net(x)

# --------- VQ with EMA codebook ---------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_codes=8192, d=256, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_codes = n_codes
        self.d = d
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(n_codes, d)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_codes, 1.0 / n_codes)

        self.register_buffer("ema_cluster_size", torch.zeros(n_codes))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    @torch.no_grad()
    def _ema_update(self, flat, codes_onehot):
        # flat: (N,D) float32, codes_onehot: (N,K) float32
        decay, eps = self.decay, self.eps

        cluster_size = codes_onehot.sum(0)  # (K,)
        self.ema_cluster_size.mul_(decay).add_(cluster_size, alpha=1 - decay)

        embed_sum = flat.t() @ codes_onehot            # (D,K)
        self.ema_w.mul_(decay).add_(embed_sum.t(), alpha=1 - decay)  # (K,D)

        n = self.ema_cluster_size.sum()
        # Normalize and avoid divide-by-zero
        cluster_size = ((self.ema_cluster_size + eps) / (n + self.n_codes * eps)) * n
        cluster_size = cluster_size + eps
        embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    def forward(self, z_e):
        # z_e: (B, D, H, W)
        B, D, H, W = z_e.shape
        flat = rearrange(z_e, "b d h w -> (b h w) d")  # (N,D)

        codebook = self.embedding.weight  # (K,D)
        # squared L2 distances
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(1)[None, :]
        )  # (N,K)

        idx = dists.argmin(dim=1)                # (N,)
        z_q = codebook[idx]                      # (N,D)
        z_q = rearrange(z_q, "(b h w) d -> b d h w", b=B, h=H, w=W)

        # The encoder commits to the selected EMA code. The codebook itself is
        # updated by EMA, so gradients must flow to z_e rather than z_q.
        commit_loss = F.mse_loss(z_e, z_q.detach())

        # EMA updates are training-only. Inference must not mutate the codebook.
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, num_classes=self.n_codes).to(flat.dtype)
                self._ema_update(flat.detach(), onehot)

        # straight-through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, idx.view(B, H, W), commit_loss

# --------- Encoder / Decoder ---------
class Encoder(nn.Module):
    def __init__(self, ch=128, stride=16):
        super().__init__()
        assert stride in (8, 16)
        s3 = 2 if stride == 16 else 1
        self.net = nn.Sequential(
            conv3(3, ch, s=2), ResBlock(ch),
            conv3(ch, ch, s=2), ResBlock(ch),
            conv3(ch, ch, s=s3), ResBlock(ch),
            nn.Conv2d(ch, 256, 1),
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, ch=128, stride=16):
        super().__init__()
        blocks = [
            nn.Conv2d(256, ch, 1), ResBlock(ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv3(ch, ch), ResBlock(ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv3(ch, ch), ResBlock(ch),
        ]
        if stride == 16:
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"), conv3(ch, ch), ResBlock(ch)]
        blocks += [nn.Conv2d(ch, 3, 1), nn.Tanh()]
        self.net = nn.Sequential(*blocks)

    def forward(self, z):
        return self.net(z)

# --------- VQ-VAE ---------
class VQVAE(nn.Module):
    def __init__(self, codebook_size=8192, code_dim=256, stride=16):
        super().__init__()
        self.stride = stride
        self.encoder = Encoder(stride=stride)
        self.pre_vq = nn.Conv2d(256, code_dim, 1)
        self.vq = VectorQuantizerEMA(n_codes=codebook_size, d=code_dim)
        self.post_vq = nn.Conv2d(code_dim, 256, 1)
        self.decoder = Decoder(stride=stride)

    def encode(self, x):
        z_e = self.pre_vq(self.encoder(x))
        _, idx, _ = self.vq(z_e)
        return idx.long()  # (B, H//s, W//s)

    def decode(self, idx):
        # idx: (B,h,w) -> z_e: (B,D,h,w)
        emb = self.vq.embedding(idx.view(-1))                # (B*h*w, D)
        z_e = emb.view(idx.shape[0], idx.shape[1], idx.shape[2], -1)
        z_e = rearrange(z_e, "b h w d -> b d h w")
        z = self.post_vq(z_e)
        return self.decoder(z)

    def forward(self, x):
        z_e = self.pre_vq(self.encoder(x))
        z_q, idx, commit = self.vq(z_e)
        z = self.post_vq(z_q)
        xhat = self.decoder(z)
        return xhat, idx, commit
