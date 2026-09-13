"""Add plain-English module headers to the public repository.

This maintenance helper is intentionally simple: it walks through the public
repo and prepends a short module-level docstring to Python files that do not
already have one. The goal is readability for supervisors, reviewers, and
non-specialist visitors who open the source code on GitHub.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FOLDER_PURPOSE = {
    "backend": "FastAPI web service and application orchestration.",
    "frontend": "Streamlit dashboard used for interactive demonstrations.",
    "semantic_ai": "Mission-specific detectors that turn images into utility maps.",
    "token_selection": "Algorithms that decide which learned image tokens are worth transmitting.",
    "transmission": "Energy and communication models for constrained satellite links.",
    "communication": "Satellite downlink analysis and mission-scale savings estimates.",
    "evaluation": "Benchmark pipelines, statistics, and publication-oriented experiments.",
    "metrics": "Research metrics such as Semantic Utility Score.",
    "baselines": "Reference codecs and baseline model wrappers used for fair comparison.",
    "deployment": "Edge-device profiling and export helpers.",
    "research": "Research objective definitions and mission framing.",
    "src": "Original VQ-VAE model and compatibility utilities.",
    "scripts": "Command-line runners for datasets, benchmarks, reports, and exports.",
    "tests": "Automated tests proving core services and research components work.",
    "datasets": "Dataset loader code only; raw imagery is intentionally not committed.",
}


def has_module_docstring(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith('"""') or stripped.startswith("'''")


def module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def header_for(path: Path) -> str:
    folder = path.relative_to(ROOT).parts[0]
    purpose = FOLDER_PURPOSE.get(folder, "Project module.")
    return (
        f'"""{module_name(path)}\n\n'
        f"Plain-English purpose: {purpose}\n\n"
        "This file is part of the public Space Semantic Compression research demo.\n"
        "It is documented at module level so researchers, supervisors, and non-specialist\n"
        "readers can understand where the file fits before reading implementation details.\n"
        '"""\n\n'
    )


def main() -> None:
    changed = 0
    for path in sorted(ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if has_module_docstring(text):
            continue
        path.write_text(header_for(path) + text, encoding="utf-8")
        changed += 1
    print(f"Added module headers to {changed} Python files.")


if __name__ == "__main__":
    main()
