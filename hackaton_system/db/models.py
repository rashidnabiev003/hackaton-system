from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Return timezone-aware UTC timestamp for SQL defaults."""

    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""


def _utcnow() -> datetime:
    """Return timezone-aware UTC timestamp for SQL defaults."""

    return datetime.now(UTC)


class Video(Base):
    __tablename__ = "videos"
    """Модель для исходных видеороликов и их метаданных."""

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    fps: Mapped[float] = mapped_column(Float, default=25.0)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    persons: Mapped[list["Person"]] = relationship(
        "Person", back_populates="video", cascade="all, delete-orphan"
    )
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="video", cascade="all, delete-orphan"
    )
    activities: Mapped[list["Activity"]] = relationship(
        "Activity", back_populates="video", cascade="all, delete-orphan"
    )
    poses: Mapped[list["PoseKeypoints"]] = relationship(
        "PoseKeypoints", back_populates="video", cascade="all, delete-orphan"
    )


class Person(Base):
    __tablename__ = "persons"
    """Уникальный человек (трек) в пределах конкретного видео."""
    __table_args__ = (
        UniqueConstraint("video_id", "track_id", name="uq_person_video_track"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    person_type: Mapped[str] = mapped_column(String(50), default="unknown")
    person_type_conf: Mapped[float] = mapped_column(Float, default=0.0)
    reid_descriptor: Mapped[str | None] = mapped_column(String, nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="persons")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="person", cascade="all, delete-orphan"
    )
    activities: Mapped[list["Activity"]] = relationship(
        "Activity", back_populates="person", cascade="all, delete-orphan"
    )
    poses: Mapped[list["PoseKeypoints"]] = relationship(
        "PoseKeypoints", back_populates="person", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"
    """Покадровые рамки для каждого человека."""
    __table_args__ = (Index("ix_detections_video_time", "video_id", "time_sec"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    x_min: Mapped[int] = mapped_column(Integer, nullable=False)
    y_min: Mapped[int] = mapped_column(Integer, nullable=False)
    x_max: Mapped[int] = mapped_column(Integer, nullable=False)
    y_max: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    video: Mapped["Video"] = relationship("Video", back_populates="detections")
    person: Mapped["Person"] = relationship("Person", back_populates="detections")


class Activity(Base):
    __tablename__ = "activities"
    """Интервалы активностей для треков."""
    __table_args__ = (Index("ix_activities_video_person", "video_id", "person_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), nullable=False)
    activity_class: Mapped[str] = mapped_column(String(50), nullable=False)
    t_start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    t_end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    activity_conf: Mapped[float] = mapped_column(Float, default=0.0)

    video: Mapped["Video"] = relationship("Video", back_populates="activities")
    person: Mapped["Person"] = relationship("Person", back_populates="activities")


class PoseKeypoints(Base):
    __tablename__ = "pose_keypoints"
    """Pose estimation results linked to tracked persons."""

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    keypoints: Mapped[list[dict[str, float]]] = mapped_column(JSON, nullable=False)
    pose_conf: Mapped[float] = mapped_column(Float, default=0.0)

    video: Mapped["Video"] = relationship("Video", back_populates="poses")
    person: Mapped["Person"] = relationship("Person", back_populates="poses")
