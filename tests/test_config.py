"""Tests for hackaton_system.config module."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def test_get_settings_reads_env(monkeypatch: Any, tmp_path: Path) -> None:
    """Settings should reflect environment overrides and use caching."""

    db_path = tmp_path / "custom.db"
    data_dir = tmp_path / "artifacts"

    monkeypatch.setenv("HACKATON_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HACKATON_DATA_DIR", str(data_dir))

    import hackaton_system.config as config

    config = importlib.reload(config)
    settings = config.get_settings()

    assert settings.database_url == f"sqlite:///{db_path}"
    assert settings.data_dir == Path(data_dir)
    # lru_cache ensures repeated calls return the same object
    assert config.get_settings() is settings


def test_pose_capture_flag_follow_env(monkeypatch: Any) -> None:
    """Pose persistence flag should follow environment overrides."""

    monkeypatch.setenv("HACKATON_ENABLE_POSE_CAPTURE", "false")

    import hackaton_system.config as config

    config = importlib.reload(config)
    settings = config.get_settings()
    assert settings.enable_pose_capture is False
