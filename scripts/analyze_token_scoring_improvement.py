"""Generate a small evidence note for the detail-aware token scoring upgrade.

This script is intentionally lightweight. It does not claim dataset-level
improvement; it creates a controlled token-grid example showing why the new
detail term keeps more boundary tokens when semantic utility is tied.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from token_selection.utility_pruner import TokenSelectionWeights, UtilityAwareTokenPruner


OUTPUT_DIR = Path("results/model_improvement_step2_token_scoring")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokens = torch.zeros((1, 16, 16), dtype=torch.long)
    utility = np.zeros((16, 16), dtype="float32")
    utility[4:12, 4:12] = 1.0

    detail = np.zeros_like(utility)
    detail[4, 4:12] = 1.0
    detail[11, 4:12] = 1.0
    detail[4:12, 4] = 1.0
    detail[4:12, 11] = 1.0

    old_pruner = UtilityAwareTokenPruner(
        TokenSelectionWeights(alpha_utility=0.65, beta_entropy=0.20, gamma_cost=0.15, delta_detail=0.0)
    )
    new_pruner = UtilityAwareTokenPruner()

    rows: list[dict[str, object]] = []
    for keep_ratio in (0.10, 0.20, 0.30, 0.40):
        for name, pruner, detail_map in (
            ("utility_entropy_only", old_pruner, None),
            ("detail_aware_utility", new_pruner, detail),
        ):
            keep, scores = pruner.select(tokens, utility, keep_ratio=keep_ratio, detail_map=detail_map)
            rows.append(
                {
                    "method": name,
                    "keep_ratio": keep_ratio,
                    "tokens_kept": int(keep.sum()),
                    "utility_mass_retained_percent": round(float(utility[keep].sum() / utility.sum() * 100.0), 2),
                    "boundary_tokens_retained": int((keep & (detail > 0)).sum()),
                    "boundary_retention_percent": round(float((keep & (detail > 0)).sum() / detail.sum() * 100.0), 2),
                    "mean_selected_score": round(float(scores[keep].mean()), 4),
                }
            )

    csv_path = OUTPUT_DIR / "token_scoring_detail_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = _write_optional_figure(utility, detail, new_pruner)
    report_path = OUTPUT_DIR / "token_scoring_improvement_report.md"
    report_path.write_text(_report_text(rows, csv_path, figure_path), encoding="utf-8")
    print(report_path)


def _write_optional_figure(utility: np.ndarray, detail: np.ndarray, pruner: UtilityAwareTokenPruner) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    tokens = torch.zeros((1, 16, 16), dtype=torch.long)
    keep, scores = pruner.select(tokens, utility, keep_ratio=0.20, detail_map=detail)
    path = OUTPUT_DIR / "detail_aware_token_selection.png"

    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.6), dpi=220)
    for axis, array, title in (
        (axes[0], utility, "Utility"),
        (axes[1], detail, "Detail"),
        (axes[2], scores, "Token score"),
        (axes[3], keep.astype("float32"), "Kept tokens"),
    ):
        axis.imshow(array, cmap="inferno", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _report_text(rows: list[dict[str, object]], csv_path: Path, figure_path: Path | None) -> str:
    lines = [
        "# Model Improvement Step 2: Detail-Aware Token Scoring",
        "",
        "## Purpose",
        "The previous utility-aware selector ranked tokens using semantic utility, token entropy, and transmission cost. This step adds a structural detail term so the selector gives extra priority to boundaries and high-frequency regions such as wildfire fronts, smoke edges, infrastructure outlines, and burn-scar contours.",
        "",
        "## What Changed",
        "- `TokenSelectionWeights` now includes `delta_detail`.",
        "- `UtilityAwareTokenPruner.select(...)` accepts an optional `detail_map`.",
        "- `SemanticService.detail_map(...)` computes a token-scale edge/detail map from the input image.",
        "- `CompressionService.compress_image(...)` passes that detail map into token selection.",
        "- Publication ablations now include `without_detail_term`.",
        "",
        "## Controlled Token-Grid Result",
        "| Method | Keep Ratio | Utility Mass Retained (%) | Boundary Tokens Retained | Boundary Retention (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['keep_ratio']} | {row['utility_mass_retained_percent']} | "
            f"{row['boundary_tokens_retained']} | {row['boundary_retention_percent']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "This controlled example isolates one model behavior: when many tokens have similar semantic utility, the detail-aware selector retains more boundary tokens. That is desirable for wildfire Earth Observation because mission-critical evidence is often located along edges, fronts, and contours rather than inside uniform regions.",
            "",
            "This is not yet a dataset-level performance claim. The next validation step is rerunning the Sentinel-2 retention benchmark and comparing `full_system` against `without_detail_term` using SUS, detector retention, PSNR, SSIM, LPIPS, and bandwidth saved.",
            "",
            f"CSV output: `{csv_path}`",
        ]
    )
    if figure_path is not None:
        lines.append(f"Figure output: `{figure_path}`")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
