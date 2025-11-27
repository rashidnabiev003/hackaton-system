"""Tests for DB session helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from sqlalchemy import text


def _reload_session(monkeypatch: Any, tmp_path: Path) -> tuple[ModuleType, Path]:
    db_path = tmp_path / "session.db"
    monkeypatch.setenv("HACKATON_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HACKATON_DATA_DIR", str(tmp_path))

    import hackaton_system.config as config
    import hackaton_system.db.session as session_module

    config = importlib.reload(config)
    session_module = importlib.reload(session_module)
    return session_module, db_path


def test_init_db_creates_sqlite_file(monkeypatch: Any, tmp_path: Path) -> None:
    session_module, db_path = _reload_session(monkeypatch, tmp_path)

    session_module.init_db()

    assert db_path.exists()


def test_get_session_provides_transaction(monkeypatch: Any, tmp_path: Path) -> None:
    session_module, _ = _reload_session(monkeypatch, tmp_path)
    session_module.init_db()

    with session_module.get_session() as session:
        result = session.execute(text("SELECT 1")).scalar_one()
        assert result == 1
