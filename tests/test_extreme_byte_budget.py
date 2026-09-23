from PIL import Image
import pytest

from evaluation.matched_rate import EncodedCandidate
from scripts.evaluate_extreme_byte_budget import choose_rdo


def test_classical_rdo_picks_best_quality_within_byte_ceiling():
    image = Image.new("RGB", (2, 2))
    choices = [
        (EncodedCandidate(bytes(90), image, 1.0), 128, 21.0),
        (EncodedCandidate(bytes(120), image, 2.0), 256, 25.0),
        (EncodedCandidate(bytes(180), image, 3.0), 512, 30.0),
    ]
    candidate, side, psnr = choose_rdo(choices, 125)
    assert candidate.size == 120
    assert side == 256
    assert psnr == 25.0


def test_classical_rdo_rejects_infeasible_budget():
    image = Image.new("RGB", (2, 2))
    with pytest.raises(RuntimeError, match="no feasible"):
        choose_rdo([(EncodedCandidate(bytes(90), image, 1.0), 128, 21.0)], 50)
