"""backend.config

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.release import DEFAULT_CHECKPOINT_PATH, DEFAULT_CHECKPOINT_SHA256


class Settings(BaseSettings):
    app_name: str = "CompressAI Semantic Compression API"
    checkpoint_path: Path = Field(default=DEFAULT_CHECKPOINT_PATH)
    checkpoint_sha256: str = Field(default=DEFAULT_CHECKPOINT_SHA256)
    output_dir: Path = Field(default=Path("outputs"))
    device: str = Field(default="auto")
    max_upload_mb: int = Field(default=25)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="COMPRESSAI_")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
