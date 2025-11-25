from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd
from sqlalchemy import Select, select

from hackaton_system.db import Activity, Detection, Person, Video, engine, init_db


@dataclass(slots=True)
class DashboardData:
    activities: pd.DataFrame
    headcount: pd.DataFrame
    person_matrix: pd.DataFrame


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

    init_db()
    activities = _load_activity_df(video_id)
    headcount = _load_headcount_df(video_id)
    person_matrix = _load_person_activity_matrix(video_id)
    return DashboardData(activities=activities, headcount=headcount, person_matrix=person_matrix)


def _load_activity_df(video_id: Optional[int]) -> pd.DataFrame:
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
    df = _load_dataframe(stmt)
    return df if not df.empty else _placeholder_activity_df()


def _load_headcount_df(video_id: Optional[int]) -> pd.DataFrame:
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
        .rename(columns={"person_id": "count"}),
    )
    return headcount


def _load_person_activity_matrix(video_id: Optional[int]) -> pd.DataFrame:
    if video_id is None:
        return _placeholder_matrix_df()
    stmt = (
        select(
            Person.person_type,
            Activity.activity_class,
            (Activity.t_end_sec - Activity.t_start_sec).label("duration"),
        )
        .join(Activity, Activity.person_id == Person.id)
        .where(Activity.video_id == video_id)
    )
    df = _load_dataframe(stmt)
    if df.empty:
        return _placeholder_matrix_df()
    df_any = cast(Any, df)
    pivot = cast(
        pd.DataFrame,
        df_any.pivot_table(
            index="person_type",
            columns="activity_class",
            values="duration",
            aggfunc="sum",
            fill_value=0,
        ).reset_index(),
    )
    return pivot


def _load_dataframe(statement: Select[Any]) -> pd.DataFrame:
    return cast(pd.DataFrame, pd.read_sql(statement, engine))


def _placeholder_activity_df() -> pd.DataFrame:
    classes = ["walking", "monitoring", "repairing", "idle"]
    starts = np.arange(0, len(classes) * 60, 60)
    ends = starts + 55
    person_types = ["mechanic", "inspector", "welder", "operator"]
    return pd.DataFrame(
        {
            "activity_class": classes,
            "t_start_sec": starts,
            "t_end_sec": ends,
            "activity_conf": np.linspace(0.6, 0.9, len(classes)),
            "person_type": person_types,
            "track_id": np.arange(1, len(classes) + 1),
        }
    )


def _placeholder_headcount_df() -> pd.DataFrame:
    seconds = np.arange(0, 300, 15)
    counts = 2 + (np.sin(seconds / 60) > 0).astype(int)
    return pd.DataFrame({"time_sec": seconds, "count": counts})


def _placeholder_matrix_df() -> pd.DataFrame:
    data: Dict[str, list[int] | list[str]] = {
        "person_type": ["mechanic", "inspector", "welder"],
        "walking": [10, 5, 2],
        "repairing": [30, 5, 40],
        "monitoring": [5, 25, 0],
        "idle": [5, 5, 5],
    }
    return pd.DataFrame(data)
