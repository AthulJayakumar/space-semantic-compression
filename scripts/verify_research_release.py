"""Check that release metadata, evidence and the optional checkpoint agree."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.release import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CHECKPOINT_SHA256,
    MODEL_ID,
    MODEL_STATUS,
    PROJECT_VERSION,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(root: Path = ROOT) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    manifest = json.loads((root / "research_release.json").read_text(encoding="utf-8"))
    model = manifest["model"]
    evidence_path = root / manifest["principal_evidence"]["report"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    proposal = (root / manifest["proposal"]).read_text(encoding="utf-8")

    expected = {
        "version": (manifest["version"], PROJECT_VERSION),
        "model ID": (model["id"], MODEL_ID),
        "model status": (model["status"], MODEL_STATUS),
        "checkpoint path": (Path(model["path"]), DEFAULT_CHECKPOINT_PATH),
        "checkpoint SHA-256": (model["sha256"], DEFAULT_CHECKPOINT_SHA256),
        "event pairs": (manifest["principal_evidence"]["event_pairs"], evidence["events"]),
        "byte ceiling": (
            manifest["principal_evidence"]["maximum_serialized_bytes"],
            evidence["byte_ceiling"],
        ),
        "sealed-test status": (
            manifest["principal_evidence"]["sealed_test_scored"],
            evidence["sealed_test_scored"],
        ),
    }
    for label, (declared, recorded) in expected.items():
        if declared != recorded:
            errors.append(f"{label} mismatch: release={declared!r}, record={recorded!r}")

    checkpoint = root / DEFAULT_CHECKPOINT_PATH
    if checkpoint.is_file():
        actual_hash = file_sha256(checkpoint)
        if checkpoint.stat().st_size != model["size_bytes"]:
            errors.append("checkpoint size does not match research_release.json")
        if actual_hash != DEFAULT_CHECKPOINT_SHA256:
            errors.append(
                f"checkpoint SHA-256 mismatch: expected {DEFAULT_CHECKPOINT_SHA256}, got {actual_hash}"
            )
    else:
        warnings.append("release checkpoint is not present; code and evidence checks only")

    required_proposal_text = (
        "95.12",
        "83.28",
        "11.85 SUS",
        "sealed and unscored",
        "does not support superiority over JPEG2000",
    )
    for text in required_proposal_text:
        if text not in proposal:
            errors.append(f"current proposal is missing required evidence text: {text!r}")

    adapted = evidence["summaries"]["Adapted VQ-VAE"]
    jpeg2000 = evidence["summaries"]["JPEG2000 RDO"]
    if adapted["burn_sus"] >= jpeg2000["burn_sus"]:
        errors.append("evidence unexpectedly contradicts the declared JPEG2000 no-go decision")
    if evidence["sealed_test_scored"]:
        errors.append("sealed test is marked as scored")

    return errors, warnings


def main() -> int:
    errors, warnings = verify_release()
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Release {PROJECT_VERSION} verified: {MODEL_ID} ({MODEL_STATUS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
