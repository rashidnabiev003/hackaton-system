"""Tests for deterministic demo seeder."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any


def _reload_db(
    monkeypatch: Any, tmp_path: Path
) -> tuple[ModuleType, ModuleType, ModuleType]:
    db_path = tmp_path / "seed.db"
    monkeypatch.setenv("HACKATON_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HACKATON_DATA_DIR", str(tmp_path))

    import hackaton_system.config as config
    import hackaton_system.db.session as session_module
    import hackaton_system.db.models as models
    import hackaton_system.db.seeder as seeder

    config = importlib.reload(config)
    session_module = importlib.reload(session_module)
    models = importlib.reload(models)
    seeder = importlib.reload(seeder)
    return seeder, session_module, models


def test_seed_demo_data_force_overwrites(monkeypatch: Any, tmp_path: Path) -> None:
    seeder, session_module, models = _reload_db(monkeypatch, tmp_path)

    inserted = seeder.seed_demo_data(force=True)
    assert inserted > 0

    skipped = seeder.seed_demo_data(force=False)
    assert skipped == 0

    with session_module.get_session() as session:
        video_count = session.query(models.Video).count()
        activity_count = session.query(models.Activity).count()

    assert video_count == 1
    assert activity_count > 0
