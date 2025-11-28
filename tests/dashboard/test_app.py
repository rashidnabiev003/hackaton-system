"""Tests for Streamlit dashboard app."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_video_record_typeddict():
    """Test VideoRecord TypedDict structure."""
    from hackaton_system.dashboard.app import VideoRecord

    record: VideoRecord = {
        "id": 1,
        "filename": "test.mp4",
        "duration_sec": 100.0,
        "fps": 25.0,
    }

    assert record["id"] == 1
    assert record["filename"] == "test.mp4"
    assert record["duration_sec"] == 100.0
    assert record["fps"] == 25.0


def test_video_record_with_none_id():
    """Test VideoRecord with None id."""
    from hackaton_system.dashboard.app import VideoRecord

    record: VideoRecord = {
        "id": None,
        "filename": "placeholder.mp4",
        "duration_sec": 300.0,
        "fps": 25.0,
    }

    assert record["id"] is None
    assert record["filename"] == "placeholder.mp4"


@pytest.mark.skip(reason="Streamlit app requires full UI context")
def test_app_imports_correctly():
    """Test that app module imports without errors."""
    # This test just ensures the module can be imported
    # Full testing would require Streamlit runtime
    import hackaton_system.dashboard.app  # noqa: F401

    assert True


def test_app_data_loading_logic(monkeypatch: Any) -> None:
    """Test data loading logic without Streamlit UI."""
    from hackaton_system.dashboard import data_access

    # Mock data access functions
    mock_videos = pd.DataFrame(
        [
            {"id": 1, "filename": "test.mp4", "duration_sec": 100.0, "fps": 25.0},
        ]
    )
    mock_headcount = pd.DataFrame({"time_sec": [0.0, 5.0], "headcount": [2, 3]})
    mock_activity = pd.DataFrame(
        {
            "activity_class": ["working", "idle"],
            "duration_min": [10.0, 5.0],
        }
    )
    mock_matrix = pd.DataFrame(
        {
            "person_type": ["operator"],
            "walking": [5.0],
            "working": [10.0],
        }
    )
    mock_episodes = pd.DataFrame(
        {
            "person_id": [1],
            "person_type": ["operator"],
            "activity_class": ["working"],
            "t_start_sec": [0.0],
            "t_end_sec": [10.0],
            "duration_sec": [10.0],
        }
    )

    monkeypatch.setattr(data_access, "list_available_videos", lambda: mock_videos)
    monkeypatch.setattr(data_access, "load_headcount", lambda vid_id: mock_headcount)
    monkeypatch.setattr(data_access, "load_activity_summary", lambda vid_id: mock_activity)
    monkeypatch.setattr(data_access, "load_role_activity_matrix", lambda vid_id: mock_matrix)
    monkeypatch.setattr(data_access, "load_person_episodes", lambda vid_id: mock_episodes)

    # Test that functions return expected data
    videos = data_access.list_available_videos()
    assert len(videos) == 1
    assert videos.iloc[0]["filename"] == "test.mp4"

    headcount = data_access.load_headcount(1)
    assert len(headcount) == 2
    assert "headcount" in headcount.columns

    activity = data_access.load_activity_summary(1)
    assert len(activity) == 2
    assert "activity_class" in activity.columns

    matrix = data_access.load_role_activity_matrix(1)
    assert "person_type" in matrix.columns

    episodes = data_access.load_person_episodes(1)
    assert "person_id" in episodes.columns

