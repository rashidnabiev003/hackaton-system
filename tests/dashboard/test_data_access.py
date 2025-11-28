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
            {"time_sec": 0.0, "headcount": 2},
            {"time_sec": 5.0, "headcount": 1},
            {"time_sec": 10.0, "headcount": 1},
        ]
    )
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_load_person_activity_matrix_builds_manual_pivot(monkeypatch: Any) -> None:
    rows: list[Row] = [
        {"person_type": "mechanic", "activity_class": "walking", "duration_sec": 10.0},
        {"person_type": "mechanic", "activity_class": "welding", "duration_sec": 5.0},
        {"person_type": "inspector", "activity_class": "walking", "duration_sec": 7.0},
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


def test_load_activity_summary_with_data(monkeypatch: Any) -> None:
    """Test load_activity_summary calculates duration_min correctly."""
    activity_df = pd.DataFrame(
        {
            "activity_class": ["walking", "walking", "idle"],
            "t_start_sec": [0.0, 60.0, 120.0],
            "t_end_sec": [30.0, 90.0, 180.0],
        }
    )

    def fake_load_activity_df(video_id: Any) -> pd.DataFrame:
        return activity_df.copy()

    monkeypatch.setattr(da, "_load_activity_df", fake_load_activity_df, raising=False)

    result = da.load_activity_summary(video_id=1)

    assert "activity_class" in result.columns
    assert "duration_min" in result.columns
    assert len(result) == 2  # walking merged, idle separate
    walking_total = result[result["activity_class"] == "walking"]["duration_min"].sum()
    assert walking_total == (30.0 + 30.0) / 60.0


def test_load_activity_summary_empty_returns_placeholder(monkeypatch: Any) -> None:
    """Test load_activity_summary returns placeholder for empty data."""
    empty_df = pd.DataFrame()

    def fake_load_activity_df(video_id: Any) -> pd.DataFrame:
        return empty_df.copy()

    monkeypatch.setattr(da, "_load_activity_df", fake_load_activity_df, raising=False)

    result = da.load_activity_summary(video_id=1)

    assert not result.empty
    assert "activity_class" in result.columns
    assert "duration_min" in result.columns


def test_load_headcount_wraps_internal_function(monkeypatch: Any) -> None:
    """Test load_headcount calls _load_headcount_df."""
    expected_df = pd.DataFrame({"time_sec": [0.0, 5.0], "headcount": [2, 3]})

    def fake_load_headcount_df(video_id: Any) -> pd.DataFrame:
        return expected_df.copy()

    monkeypatch.setattr(da, "_load_headcount_df", fake_load_headcount_df, raising=False)

    result = da.load_headcount(video_id=1)

    pd.testing.assert_frame_equal(result, expected_df)


def test_load_role_activity_matrix_converts_to_minutes(monkeypatch: Any) -> None:
    """Test load_role_activity_matrix converts seconds to minutes."""
    matrix_df = pd.DataFrame(
        {
            "person_type": ["mechanic", "inspector"],
            "walking": [120.0, 60.0],  # seconds
            "welding": [180.0, 0.0],
        }
    )

    def fake_load_person_activity_matrix(video_id: Any) -> pd.DataFrame:
        return matrix_df.copy()

    monkeypatch.setattr(
        da, "_load_person_activity_matrix", fake_load_person_activity_matrix, raising=False
    )

    result = da.load_role_activity_matrix(video_id=1)

    assert result["walking"].iloc[0] == 2.0  # 120 seconds = 2 minutes
    assert result["walking"].iloc[1] == 1.0  # 60 seconds = 1 minute
    assert result["welding"].iloc[0] == 3.0  # 180 seconds = 3 minutes


def test_load_role_activity_matrix_empty_returns_placeholder(monkeypatch: Any) -> None:
    """Test load_role_activity_matrix returns placeholder for empty data."""
    empty_df = pd.DataFrame()

    def fake_load_person_activity_matrix(video_id: Any) -> pd.DataFrame:
        return empty_df.copy()

    monkeypatch.setattr(
        da, "_load_person_activity_matrix", fake_load_person_activity_matrix, raising=False
    )

    result = da.load_role_activity_matrix(video_id=1)

    assert not result.empty
    assert "person_type" in result.columns


def test_load_person_episodes_with_none_returns_placeholder() -> None:
    """Test load_person_episodes returns placeholder for None video_id."""
    result = da.load_person_episodes(video_id=None)

    assert not result.empty
    assert "person_id" in result.columns
    assert "activity_class" in result.columns


def test_load_person_episodes_with_data(monkeypatch: Any) -> None:
    """Test load_person_episodes loads episodes from database."""
    episodes_df = pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "person_type": ["operator", "operator", "supervisor"],
            "activity_class": ["working", "idle_at_station", "walking"],
            "t_start_sec": [0.0, 60.0, 120.0],
            "t_end_sec": [30.0, 90.0, 180.0],
            "duration_sec": [30.0, 30.0, 60.0],
        }
    )

    def fake_load_dataframe(stmt: Any) -> pd.DataFrame:
        return episodes_df.copy()

    monkeypatch.setattr(da, "_load_dataframe", fake_load_dataframe, raising=False)

    result = da.load_person_episodes(video_id=1)

    assert len(result) == 3
    assert "person_id" in result.columns
    assert "activity_class" in result.columns
    assert "duration_sec" in result.columns


def test_load_person_episodes_empty_returns_placeholder(monkeypatch: Any) -> None:
    """Test load_person_episodes returns placeholder for empty data."""
    empty_df = pd.DataFrame()

    def fake_load_dataframe(stmt: Any) -> pd.DataFrame:
        return empty_df.copy()

    monkeypatch.setattr(da, "_load_dataframe", fake_load_dataframe, raising=False)

    result = da.load_person_episodes(video_id=1)

    assert not result.empty
    assert "person_id" in result.columns
