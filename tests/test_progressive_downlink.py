"""Progressive stream decoding uses complete received records only."""

import numpy as np
import pytest
import torch

from evaluation.progressive_downlink import HEADER, RECORD, decode_prefix, paired_comparisons, serialize_progressive


def test_prefix_decoding_and_byte_accounting() -> None:
    tokens = torch.tensor([[[3, 7], [3, 9]]])
    stream = serialize_progressive(tokens, np.array([3, 1, 0, 2]))
    received, used = decode_prefix(stream, HEADER.size + RECORD.size * 2 + 1)
    assert used == HEADER.size + RECORD.size * 2
    assert received.tolist() == [[[3, 7], [3, 9]]]
    full, used = decode_prefix(stream, len(stream))
    assert torch.equal(full, tokens)
    assert used == len(stream)


def test_truncated_header_and_invalid_ranking() -> None:
    tokens = torch.tensor([[[1, 2]]])
    with pytest.raises(ValueError, match="Ranking"):
        serialize_progressive(tokens, np.array([0, 0]))
    stream = serialize_progressive(tokens, np.array([0, 1]))
    with pytest.raises(ValueError, match="header"):
        decode_prefix(stream, HEADER.size - 1)
    with pytest.raises(ValueError, match="one image"):
        serialize_progressive(torch.cat((tokens, tokens)), np.array([0, 1]))


def test_progressive_comparison_counts_event_groups() -> None:
    rows = []
    for index in range(3):
        for chip in range(2):
            for method, sus in (("vqvae_random", 50), ("vqvae_utility", 60 + index)):
                rows.append({
                    "sample_id": f"{index}-{chip}",
                    "event_group": f"event-{index}",
                    "method": method,
                    "retention": 0.2,
                    "sus": sus,
                    "detector_retention": 0.5,
                })
    result = next(item for item in paired_comparisons(rows) if item["metric"] == "sus")
    assert result["n_events"] == 3
    assert result["mean_difference"] == 11
