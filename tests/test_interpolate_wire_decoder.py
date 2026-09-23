"""Guard checkpoint compatibility when testing a decoder midpoint."""

from __future__ import annotations

import torch

from scripts.interpolate_wire_decoder import interpolate


def test_midpoint_changes_only_decoder_weights(tmp_path):
    base = {"model": {"encoder.weight": torch.tensor([1.0]), "decoder.weight": torch.tensor([2.0])}, "args": {"stride": 16}}
    task = {"model": {"encoder.weight": torch.tensor([1.0]), "decoder.weight": torch.tensor([4.0])}, "args": {"stride": 16}}
    base_path, task_path = tmp_path / "base.pt", tmp_path / "task.pt"
    torch.save(base, base_path)
    torch.save(task, task_path)
    result = interpolate(base_path, task_path, 0.5)
    assert result["model"]["encoder.weight"].item() == 1.0
    assert result["model"]["decoder.weight"].item() == 3.0
