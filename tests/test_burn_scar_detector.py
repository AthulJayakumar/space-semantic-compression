"""Tests for the supervised burn-scar utility detector."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from semantic_ai.burn_scar_model import BurnScarUtilityNet
from semantic_ai.wildfire_detector import WildfireDetector


def test_burn_scar_model_preserves_spatial_shape():
    model = BurnScarUtilityNet(base_channels=4)
    output = model(torch.rand(2, 3, 32, 32))
    assert output.shape == (2, 1, 32, 32)


def test_wildfire_detector_loads_supervised_checkpoint(tmp_path):
    model = BurnScarUtilityNet(base_channels=4)
    checkpoint = tmp_path / "burn_scar.pt"
    torch.save(
        {
            "model_type": "burn_scar_utility_unet",
            "model": model.state_dict(),
            "config": {"base_channels": 4, "input_size": 32, "threshold": 0.5},
        },
        checkpoint,
    )
    detector = WildfireDetector(checkpoint, output_dir=None, save_visualizations=False)
    output = detector.detect(Image.new("RGB", (48, 40), (120, 80, 60)))
    assert output.backend == "supervised_burn_scar_unet"
    assert output.utility_map.shape == (40, 48)
    assert np.all((output.utility_map >= 0.0) & (output.utility_map <= 1.0))
