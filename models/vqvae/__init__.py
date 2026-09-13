"""models.vqvae.__init__

Plain-English purpose: Project module.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from src.models.vqvae import Decoder, Encoder, VQVAE, VectorQuantizerEMA

__all__ = ["Decoder", "Encoder", "VQVAE", "VectorQuantizerEMA"]
