from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized project configuration loaded from environment variables."""

    # Используем модельную конфигурацию Pydantic, чтобы подтягивать значения из .env
    # c префиксом HACKATON_. Эта секция по сути описывает «откуда» приходят данные.
    model_config = SettingsConfigDict(
        env_prefix="HACKATON_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # URL подключения к БД; по умолчанию используем локальный SQLite в каталоге data.
    database_url: str = Field(
        default="sqlite:///data/hackaton.db",
        description="SQLAlchemy-compatible database URL.",
    )
    # Путь к каталогу с данными — через него создаём директории при инициализации БД.
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for processed artifacts and SQLite database.",
    )
    # Опциональные пути к весам: оставляем None, чтобы брать значения по умолчанию из Ultralytics.
    detection_model_path: str | None = Field(
        default=None, description="Path to the YOLO weights file."
    )
    activity_model_path: str | None = Field(
        default=None, description="Path to the action recognition weights."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""

    # lru_cache гарантирует, что настройки считываются один раз за запуск процесса,
    # а дальнейшие импорты получают готовый объект без повторного чтения .env.
    return Settings()
