import numpy as np
import torch

from backend.services.token_transmission_service import TokenTransmissionService
from scripts.finetune_wire_vqvae import received_tokens


def test_wire_finetuning_replaces_missing_codes_with_selected_mode():
    tokens = torch.tensor([[[9, 2], [2, 9]]])
    ranking = np.array([0, 3, 1, 2])
    result = received_tokens(tokens, ranking, 0.5)
    assert result.tolist() == [[[9, 9], [9, 9]]]


def test_wire_finetuning_preserves_selected_positions():
    tokens = torch.tensor([[[9, 2], [4, 7]]])
    ranking = np.array([0, 2, 1, 3])
    result = received_tokens(tokens, ranking, 0.5)
    assert result[0, 0, 0] == 9
    assert result[0, 1, 0] == 4


def test_wire_finetuning_matches_serialized_payload_receiver():
    service = TokenTransmissionService()
    tokens = torch.tensor([[[4, 9, 4], [7, 9, 1]]])
    ranking = np.array([0, 2, 4, 1, 3, 5])
    mask = np.zeros(6, dtype=bool)
    mask[ranking[:3]] = True
    wire, _ = service.deserialize_payload(service.serialize_payload(tokens, mask.reshape(2, 3)))
    torch.testing.assert_close(received_tokens(tokens, ranking, 0.5), wire)
