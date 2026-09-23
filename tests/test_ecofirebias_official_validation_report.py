"""Guard official-validation reporting against split, pairing and wire errors."""

from __future__ import annotations

import pytest

from scripts.report_ecofirebias_official_validation import validate_rows


def fixture_rows():
    plan = {"sample_ids": ["burn-1", "neg-1"], "maximum_wire_bytes_per_image": 1200}
    manifest = {
        sample_id: {"event_group": "event-1", "label_source": "quantized_dnbr_gt_027_proxy"}
        for sample_id in plan["sample_ids"]
    }
    kinds = {"burn-1": "burn", "neg-1": "neg"}
    rows = [
        {"sample_id": sample_id, "method": method, "event_group": "event-1",
         "source_split": "val", "split": "official_validation",
         "label_source": "quantized_dnbr_gt_027_proxy", "target_bytes": "1200",
         "encoded_bytes": "1198", "sus": "80", "label_dice": "0.4"}
        for method in ("vqvae_fixed_utility", "jpeg_rdo", "jpeg2000_rdo")
        for sample_id in plan["sample_ids"]
    ]
    return rows, plan, manifest, kinds


def test_accepts_complete_frozen_paired_rows() -> None:
    rows, plan, manifest, kinds = fixture_rows()
    indexed = validate_rows(rows, plan, manifest, kinds)
    assert all(len(method_rows) == 2 for method_rows in indexed.values())


@pytest.mark.parametrize("field,value", [
    ("encoded_bytes", "1201"),
    ("source_split", "test"),
    ("event_group", "other-event"),
    ("label_source", "invented"),
])
def test_rejects_out_of_plan_rows(field: str, value: str) -> None:
    rows, plan, manifest, kinds = fixture_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_rows(rows, plan, manifest, kinds)


def test_rejects_missing_method_sample() -> None:
    rows, plan, manifest, kinds = fixture_rows()
    with pytest.raises(ValueError):
        validate_rows(rows[:-1], plan, manifest, kinds)
