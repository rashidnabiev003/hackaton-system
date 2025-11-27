# pyright: reportPrivateUsage=false
"""Tests for dashboard data access utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hackaton_system.dashboard import data_access as da


RowValue = float | int | str
Row = dict[str, RowValue]


def test_load_headcount_df_groups_unique_people(monkeypatch: Any) -> None:
    rows: list[Row] = [
        {"time_sec": 0.0, "person_id": 1},
        {"time_sec": 0.0, "person_id": 2},
        {"time_sec": 5.0, "person_id": 2},
        {"time_sec": 5.0, "person_id": 2},
        {"time_sec": 10.0, "person_id": 3},
    ]
    def fake_fetch_rows(stmt: Any) -> list[Row]:
        return rows.copy()

    monkeypatch.setattr(da, "_fetch_rows", fake_fetch_rows, raising=False)

    df: pd.DataFrame = da._load_headcount_df(video_id=1)

    expected = pd.DataFrame(
        [
            {"time_sec": 0.0, "count": 2},
            {"time_sec": 5.0, "count": 1},
            {"time_sec": 10.0, "count": 1},
        ]
    )
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_load_person_activity_matrix_builds_manual_pivot(monkeypatch: Any) -> None:
    rows: list[Row] = [
        {"person_type": "mechanic", "activity_class": "walking", "duration": 10.0},
        {"person_type": "mechanic", "activity_class": "welding", "duration": 5.0},
        {"person_type": "inspector", "activity_class": "walking", "duration": 7.0},
    ]
    def fake_fetch_rows(stmt: Any) -> list[Row]:
        return rows.copy()

    monkeypatch.setattr(da, "_fetch_rows", fake_fetch_rows, raising=False)

    df: pd.DataFrame = da._load_person_activity_matrix(video_id=42)

    assert set(df.columns) == {"person_type", "walking", "welding"}
    mechanic = df[df["person_type"] == "mechanic"].iloc[0]
    inspector = df[df["person_type"] == "inspector"].iloc[0]
    assert mechanic["walking"] == 10.0
    assert mechanic["welding"] == 5.0
    assert inspector["walking"] == 7.0
    assert inspector["welding"] == 0.0


def test_load_activity_df_falls_back_to_placeholder(monkeypatch: Any) -> None:
    empty_df = pd.DataFrame()

    def fake_loader(statement: Any) -> pd.DataFrame:
        return empty_df

    monkeypatch.setattr(da, "_load_dataframe", fake_loader, raising=False)

    result: pd.DataFrame = da._load_activity_df(video_id=7)

    assert result.equals(da._placeholder_activity_df())
