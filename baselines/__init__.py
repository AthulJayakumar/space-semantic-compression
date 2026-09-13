"""baselines.__init__

Plain-English purpose: Reference codecs and baseline model wrappers used for fair comparison.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from baselines.neural_codecs import AutoencoderCodec, LightweightCNNCodec, VariationalAutoencoderCodec

__all__ = ["AutoencoderCodec", "VariationalAutoencoderCodec", "LightweightCNNCodec"]
