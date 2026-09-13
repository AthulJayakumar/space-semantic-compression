"""tests.test_roundtrip

Plain-English purpose: Automated tests proving core services and research components work.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

import torch
from src.models.vqvae import VQVAE

def test_encode_decode_shapes():
    m = VQVAE(codebook_size=64)
    x = torch.randn(2,3,256,256)
    idx = m.encode(x)
    xhat = m.decode(idx)
    assert xhat.shape == x.shape


def test_eval_encode_does_not_mutate_codebook():
    m = VQVAE(codebook_size=64).eval()
    before = m.vq.embedding.weight.detach().clone()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        _ = m.encode(x)

    assert torch.equal(before, m.vq.embedding.weight.detach())
