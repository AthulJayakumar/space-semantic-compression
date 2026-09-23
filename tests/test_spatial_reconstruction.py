from io import BytesIO

import numpy as np
import pytest
import torch

from backend.services.token_transmission_service import TokenTransmissionService
from token_selection.spatial_reconstruction import decode_nearest_filled_tokens, decode_spatially_filled_tokens


def test_sparse_payload_roundtrip_uses_received_codes_only():
    service = TokenTransmissionService()
    tokens = torch.tensor([[[4, 7], [7, 7]]])
    mask = np.array([[True, False], [False, False]])
    payload = service.serialize_payload(tokens, mask)
    restored, restored_mask = service.deserialize_payload(payload)
    assert restored.tolist() == [[[4, 4], [4, 4]]]
    np.testing.assert_array_equal(restored_mask, mask)


def test_payload_rejects_mismatched_mask_and_codes():
    service = TokenTransmissionService()
    buffer = BytesIO()
    np.savez_compressed(buffer, codes=np.array([2], dtype=np.uint16), mask=np.array([[1, 1]], dtype=np.uint8))
    with pytest.raises(ValueError, match="count"):
        service.deserialize_payload(buffer.getvalue())


def test_spatial_fill_preserves_received_embeddings():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vq = torch.nn.Module()
            self.vq.embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[0.0], [2.0], [4.0]]))
            self.post_vq = torch.nn.Identity()
            self.decoder = torch.nn.Identity()

    received = torch.tensor([[[0, 0], [0, 2]]])
    mask = np.array([[True, False], [False, True]])
    output = decode_spatially_filled_tokens(Dummy(), received, mask)
    assert output.shape == (1, 1, 2, 2)
    assert output[0, 0, 0, 0].item() == 0.0
    assert output[0, 0, 1, 1].item() == 4.0
    assert output[0, 0, 0, 1].item() == pytest.approx(2.0)


def test_nearest_fill_uses_only_received_codes():
    class Dummy(torch.nn.Module):
        def decode(self, codes):
            return codes

    received = torch.tensor([[[1, 0, 0, 2]]])
    mask = np.array([[True, False, False, True]])
    output = decode_nearest_filled_tokens(Dummy(), received, mask)
    assert output.tolist() == [[[1, 1, 2, 2]]]
