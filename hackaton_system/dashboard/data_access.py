from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd
from sqlalchemy import Select, select

from hackaton_system.db import Activity, Detection, Person, Video, engine, init_db


@dataclass(slots=True)
class DashboardData:
    headcount: pd.DataFrame
    activity_summary: pd.DataFrame
    role_activity_matrix: pd.DataFrame
    person_episodes: pd.DataFrame


def list_available_videos() -> pd.DataFrame:
    """Возвращает список обработанных видео или заглушку."""

    init_db()
    stmt = select(Video.id, Video.filename, Video.duration_sec, Video.fps).order_by(Video.id)
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

    return DashboardData(
        headcount=load_headcount(video_id),
        activity_summary=load_activity_summary(video_id),
        role_activity_matrix=load_role_activity_matrix(video_id),
        person_episodes=load_person_episodes(video_id),
    )


def load_headcount(video_id: Optional[int]) -> pd.DataFrame:
    """Возвращает количество людей по кадрам."""

    init_db()
    if video_id is None:
        return _placeholder_headcount_df()
    stmt = (
        select(Detection.time_sec.label("time_sec"), Detection.person_id)
        .where(Detection.video_id == video_id)
        .order_by(Detection.time_sec)
    )
    df = _load_dataframe(stmt)
    if df.empty:
        return _placeholder_headcount_df()
    df_any = cast(Any, df)
    headcount = cast(
        pd.DataFrame,
        df_any.groupby("time_sec")["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "headcount"}),
    )
    return headcount


def load_activity_summary(video_id: Optional[int]) -> pd.DataFrame:
    """Суммарное время по активностям, минуты."""

    init_db()
    if video_id is None:
        return _placeholder_activity_summary_df()
    stmt = (
        select(
            Activity.activity_class,
            (Activity.t_end_sec - Activity.t_start_sec).label("duration_sec"),
        )
        .where(Activity.video_id == video_id)
    )
    df = _load_dataframe(stmt)
    if df.empty:
        return _placeholder_activity_summary_df()
    df_any = cast(Any, df)
    summary = cast(
        pd.DataFrame,
        df_any.groupby("activity_class")["duration_sec"].sum().reset_index(),
    )
    summary["duration_min"] = summary["duration_sec"] / 60.0
    return summary.sort_values("duration_min", ascending=False).reset_index(drop=True)


def load_role_activity_matrix(video_id: Optional[int]) -> pd.DataFrame:
    """Матрица время(минуты) по person_type × activity."""

    init_db()
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
    df = _load_dataframe(stmt)
    if df.empty:
        return _placeholder_matrix_df()
    df_any = cast(Any, df)
    df_any["duration_min"] = df_any["duration_sec"] / 60.0
    pivot = cast(
        pd.DataFrame,
        df_any.pivot_table(
            index="person_type",
            columns="activity_class",
            values="duration_min",
            aggfunc="sum",
            fill_value=0,
        ).reset_index(),
    )
    pivot = pivot.sort_values("person_type").reset_index(drop=True)
    return pivot


def load_person_episodes(video_id: Optional[int]) -> pd.DataFrame:
    """Детальные эпизоды по людям."""

    init_db()
    if video_id is None:
        return _placeholder_episodes_df()
    stmt = (
        select(
            Activity.person_id,
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


def _load_dataframe(statement: Select[Any]) -> pd.DataFrame:
    return cast(pd.DataFrame, pd.read_sql(statement, engine))


def _placeholder_activity_summary_df() -> pd.DataFrame:
    classes = ["working", "idle_at_station", "walking", "standing", "in_restricted_zone"]
    mins = np.array([18, 7, 5, 3, 1], dtype=float)
    return pd.DataFrame(
        {
            "activity_class": classes,
            "duration_sec": mins * 60,
            "duration_min": mins,
        }
    )


def _placeholder_headcount_df() -> pd.DataFrame:
    seconds = np.arange(0, 300, 15)
    counts = 2 + (np.sin(seconds / 60) > 0).astype(int)
    return pd.DataFrame({"time_sec": seconds, "headcount": counts})


def _placeholder_matrix_df() -> pd.DataFrame:
    data: Dict[str, list[float] | list[str]] = {
        "person_type": ["operator", "supervisor", "visitor"],
        "working": [18.0, 1.0, 0.0],
        "idle_at_station": [6.5, 0.5, 0.0],
        "walking": [2.0, 6.0, 4.0],
        "standing": [0.5, 4.0, 3.0],
        "in_restricted_zone": [0.0, 0.0, 1.5],
    }
    return pd.DataFrame(data)


def _placeholder_episodes_df() -> pd.DataFrame:
    starts = np.arange(0, 240, 60, dtype=float)
    ends = starts + 50
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "person_type": ["operator", "operator", "supervisor", "visitor"],
            "activity_class": ["working", "idle_at_station", "walking", "in_restricted_zone"],
            "t_start_sec": starts[:4],
            "t_end_sec": ends[:4],
            "duration_sec": ends[:4] - starts[:4],
        }
    )
