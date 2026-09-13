"""backend.utils.file_utils

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from pathlib import Path
from uuid import uuid4


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_output_path(output_dir: Path, suffix: str = ".png") -> Path:
    ensure_dir(output_dir)
    return output_dir / f"reconstruction_{uuid4().hex}{suffix}"


def file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0
