from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized project configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="HACKATON_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    database_url: str = Field(
        default="sqlite:///data/hackaton.db",
        description="SQLAlchemy-compatible database URL.",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for processed artifacts and SQLite database.",
    )
    detection_model_path: str | None = Field(
        default=None, description="Path to the YOLO weights file."
    )
    activity_model_path: str | None = Field(
        default=None, description="Path to the action recognition weights."
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""

    return Settings()
