from pathlib import Path

from scripts.verify_research_release import verify_release
from src.release import MODEL_STATUS, PROJECT_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_research_release_metadata_is_consistent() -> None:
    errors, _warnings = verify_release(ROOT)
    assert errors == []


def test_release_is_explicitly_experimental() -> None:
    assert PROJECT_VERSION == "0.2.0"
    assert MODEL_STATUS == "experimental_no_go_for_sealed_test"
