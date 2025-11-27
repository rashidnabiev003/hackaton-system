from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from hackaton_system.config import get_settings
from hackaton_system.db.models import Base

# Настройки читаем один раз, чтобы не открывать .env в каждом модуле.
settings = get_settings()

# Engine и фабрика сессий создаются при импорте — приложение использует их повторно.
engine: Engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create database tables if they do not yet exist."""

    if settings.database_url.startswith("sqlite:///"):
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    # DeclarativeBase знает про все модели, поэтому create_all создаёт таблицы целиком.
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """Provide transactional scope for DB operations."""

    session = SessionLocal()
    try:
        # Возвращаем сессию вызывающему коду; commit/rollback управляются здесь.
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
