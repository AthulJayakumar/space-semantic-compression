"""Acquire and freeze the FireScope cross-dataset evaluation lockbox."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.external_lockbox import (  # noqa: E402
    DEFAULT_QUOTAS,
    FIRESCOPE_REPO_ID,
    acquire_firescope_selection,
    audit_and_freeze_lockbox,
    download_and_prepare_lockbox,
    pair_firescope_files,
    select_lockbox_candidates,
    FireScopeCandidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen 100-scene FireScope external lockbox.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/external_lockbox/firescope_100"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_lockbox_firescope_100"))
    parser.add_argument(
        "--revision",
        default="7f57fb51b22733cb021794066d3fd8675bb30d83",
        help="Pinned FireScope repository revision.",
    )
    parser.add_argument(
        "--index-repo",
        type=Path,
        default=Path("datasets/external_lockbox/firescope_index_repo"),
        help="Optional metadata-only Git checkout used when the Hub API is rate-limited.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before preparing the external lockbox") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "lockbox_selection_plan.json"
    revision = args.revision
    if selection_path.exists():
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_payload["source_revision"] != revision:
            raise ValueError("Cached lockbox selection revision does not match --revision")
        selected = [FireScopeCandidate(**row) for row in selection_payload["selected"]]
        candidate_count = int(selection_payload["paired_candidates"])
    else:
        if (args.index_repo / ".git").exists():
            indexed_revision = subprocess.run(
                ["git", "-C", str(args.index_repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if indexed_revision != revision:
                raise ValueError("Metadata checkout revision does not match --revision")
            paths = subprocess.run(
                ["git", "-C", str(args.index_repo), "ls-tree", "-r", "--name-only", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        else:
            api = HfApi()
            paths = api.list_repo_files(FIRESCOPE_REPO_ID, repo_type="dataset", revision=revision)
        candidates = pair_firescope_files(paths)
        selected = select_lockbox_candidates(candidates)
        candidate_count = len(candidates)
        selection_path.write_text(
            json.dumps(
                {
                    "source_repository": FIRESCOPE_REPO_ID,
                    "source_revision": revision,
                    "selection_quotas": DEFAULT_QUOTAS,
                    "paired_candidates": candidate_count,
                    "selected": [asdict(candidate) for candidate in selected],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    acquire_firescope_selection(selected, args.data_root, revision=revision)
    rows = download_and_prepare_lockbox(selected, args.data_root, revision=revision)

    manifest_path = args.output_dir / "firescope_100_lockbox_manifest.csv"
    freeze_path = args.output_dir / "LOCKBOX_FROZEN.json"
    checkpoint_paths = [
        Path("models/checkpoints/vqvae_s16k8_mixed_regularized_finetuned.pt"),
        Path("models/checkpoints/wildfire_utility_segmentation_retrained.pt"),
    ]
    freeze = audit_and_freeze_lockbox(
        rows,
        manifest_path,
        freeze_path,
        source_revision=revision,
        repository_root=ROOT,
        checkpoint_paths=checkpoint_paths,
        expected_count=sum(DEFAULT_QUOTAS.values()),
    )
    summary = {
        "paired_candidates": candidate_count,
        "selected_scenes": len(selected),
        "manifest": str(manifest_path),
        "freeze_record": str(freeze_path),
        "manifest_sha256": freeze["manifest_sha256"],
        "audit_checks": freeze["audit_checks"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
