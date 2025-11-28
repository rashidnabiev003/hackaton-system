from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import Select, select

from hackaton_system.db.models import Activity, Detection, Person, Video
from hackaton_system.db.session import get_session, init_db


@dataclass(slots=True)
class DashboardData:
    activities: DataFrame
    headcount: DataFrame
    person_matrix: DataFrame


def list_available_videos() -> DataFrame:
    """Возвращает список обработанных видео или заглушку."""

    # Инициализируем БД на случай запуска сервиса «с нуля».
    init_db()
    stmt = select(Video.id, Video.filename, Video.duration_sec, Video.fps).order_by(
        Video.id
    )
    df = _load_dataframe(stmt)
    if not df.empty:
        return df
    return pd.DataFrame(
        [
            {
                "id": None,
                "filename": "Demo placeholder",
                "duration_sec": 300,
                "fps": 25.0,
            }
        ]
    )


def load_dashboard_data(video_id: Optional[int]) -> DashboardData:
    """Собирает набор датафреймов для отрисовки панели."""

    # Гарантируем актуальную схему БД и собираем все срезы отчётности.
    init_db()
    activities = _load_activity_df(video_id)
    headcount = _load_headcount_df(video_id)
    person_matrix = _load_person_activity_matrix(video_id)
    return DashboardData(
        activities=activities, headcount=headcount, person_matrix=person_matrix
    )


def _load_activity_df(video_id: Optional[int]) -> DataFrame:
    if video_id is None:
        return _placeholder_activity_df()
    stmt = (
        select(
            Activity.id.label("activity_id"),
            Activity.activity_class,
            Activity.t_start_sec,
            Activity.t_end_sec,
            Activity.activity_conf,
            Person.person_type,
            Person.track_id,
        )
        .join(Person, Activity.person_id == Person.id)
        .where(Activity.video_id == video_id)
        .order_by(Activity.t_start_sec)
    )
    df: DataFrame = _load_dataframe(stmt)
    return df if not df.empty else _placeholder_activity_df()


def _load_headcount_df(video_id: Optional[int]) -> DataFrame:
    if video_id is None:
        return _placeholder_headcount_df()
    stmt = (
        select(Detection.time_sec.label("time_sec"), Detection.person_id)
        .where(Detection.video_id == video_id)
        .order_by(Detection.time_sec)
    )
    rows = _fetch_rows(stmt)
    if not rows:
        return _placeholder_headcount_df()
    # Считаем уникальные track_id внутри каждого момента времени.
    counts: Dict[float, set[int]] = defaultdict(set)
    for row in rows:
        counts[float(row["time_sec"])].add(int(row["person_id"]))
    times = sorted(counts.keys())
    # Преобразуем интервальные множества в плоскую таблицу для построения графика.
    data: Dict[str, list[float] | list[int]] = {
        "time_sec": times,
        "headcount": [len(counts[time_sec]) for time_sec in times],
    }
    return pd.DataFrame(data)


def _load_person_activity_matrix(video_id: Optional[int]) -> DataFrame:
    if video_id is None:
        return _placeholder_matrix_df()
    stmt = (
        select(
            Person.person_type,
            Activity.activity_class,
            (Activity.t_end_sec - Activity.t_start_sec).label("duration_sec"),
        )
        .join(Person, Activity.person_id == Person.id)
        .where(Activity.video_id == video_id)
    )
    rows = _fetch_rows(stmt)
    if not rows:
        return _placeholder_matrix_df()
    # Накопим длительности в структуре {person_type -> {activity_class -> duration}}.
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_classes: set[str] = set()
    for row in rows:
        person_type = str(row["person_type"])
        activity_class = str(row["activity_class"])
        duration = float(row["duration_sec"])
        totals[person_type][activity_class] += duration
        all_classes.add(activity_class)
    sorted_classes = sorted(all_classes)
    records: list[dict[str, Any]] = []
    for person_type in sorted(totals.keys()):
        record: dict[str, Any] = {"person_type": person_type}
        for activity_class in sorted_classes:
            record[activity_class] = totals[person_type].get(activity_class, 0.0)
        records.append(record)
    return pd.DataFrame(records)


def _load_dataframe(statement: Select[Any]) -> DataFrame:
    rows = _fetch_rows(statement)
    return pd.DataFrame(rows)


def _fetch_rows(statement: Select[Any]) -> list[dict[str, Any]]:
    # Выполняем SQL-запрос в отдельной сессии и приводим строки к dict для Pandas.
    with get_session() as session:
        result = session.execute(statement)
        return [dict(row) for row in result.mappings().all()]


def _placeholder_activity_df() -> DataFrame:
    classes = ["walking", "monitoring", "repairing", "idle"]
    duration_min = [10.0, 5.0, 30.0, 15.0]
    return pd.DataFrame(
        {
            "activity_class": classes,
            "duration_min": duration_min,
        }
    )


def _placeholder_headcount_df() -> DataFrame:
    seconds = np.arange(0, 300, 15)
    counts = 2 + (np.sin(seconds / 60) > 0).astype(int)
    return pd.DataFrame({"time_sec": seconds, "headcount": counts})


def _placeholder_matrix_df() -> DataFrame:
    data: Dict[str, list[int] | list[str]] = {
        "person_type": ["mechanic", "inspector", "welder"],
        "walking": [10, 5, 2],
        "repairing": [30, 5, 40],
        "monitoring": [5, 25, 0],
        "idle": [5, 5, 5],
    }
    return pd.DataFrame(data)


def _placeholder_episodes_df() -> pd.DataFrame:
    starts = np.arange(0, 240, 60, dtype=float)
    ends = starts + 50
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "track_id": [101, 101, 202, 303],
            "person_type": ["operator", "operator", "supervisor", "visitor"],
            "activity_class": [
                "working",
                "idle_at_station",
                "walking",
                "in_restricted_zone",
            ],
            "t_start_sec": starts[:4],
            "t_end_sec": ends[:4],
            "duration_sec": ends[:4] - starts[:4],
        }
    )


def load_activity_summary(video_id: Optional[int]) -> DataFrame:
    """Load activity summary with duration in minutes."""
    df = _load_activity_df(video_id)
    if df.empty:
        return _placeholder_activity_df()
    # Calculate duration in minutes from t_start_sec and t_end_sec
    if "t_start_sec" in df.columns and "t_end_sec" in df.columns:
        df = df.copy()
        df["duration_sec"] = df["t_end_sec"] - df["t_start_sec"]
        df["duration_min"] = df["duration_sec"] / 60.0
        # Group by activity_class and sum durations
        summary = df.groupby("activity_class", as_index=False)["duration_min"].sum()
        return summary
    # If already has duration_min, return as is
    if "duration_min" in df.columns:
        return df.groupby("activity_class", as_index=False)["duration_min"].sum()
    return _placeholder_activity_df()


def load_headcount(video_id: Optional[int]) -> DataFrame:
    """Load headcount over time."""
    return _load_headcount_df(video_id)


def load_role_activity_matrix(video_id: Optional[int]) -> DataFrame:
    """Load person type × activity matrix with durations in minutes."""
    df = _load_person_activity_matrix(video_id)
    if df.empty:
        return _placeholder_matrix_df()
    # Convert all numeric columns (activities) from seconds to minutes
    df = df.copy()
    activity_cols = [col for col in df.columns if col != "person_type"]
    for col in activity_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] / 60.0
    return df


def load_person_episodes(video_id: Optional[int]) -> DataFrame:
    """Load person activity episodes."""
    if video_id is None:
        return _placeholder_episodes_df()
    stmt = (
        select(
            Person.id.label("person_id"),
            Person.track_id,
            Person.person_type,
            Activity.activity_class,
            Activity.t_start_sec,
            Activity.t_end_sec,
            (Activity.t_end_sec - Activity.t_start_sec).label("duration_sec"),
        )
        .join(Person, Activity.person_id == Person.id)
        .where(Activity.video_id == video_id)
        .order_by(Activity.t_start_sec)
    )
    df = _load_dataframe(stmt)
    return df if not df.empty else _placeholder_episodes_df()
