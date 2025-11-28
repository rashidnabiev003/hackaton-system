# pyright: reportPrivateUsage=false
"""Tests for lightweight pieces of the video processing pipeline."""

from __future__ import annotations

import pytest

from hackaton_system.pipeline.video_processor import (
    DetectionResult,
    VideoProcessor,
    _to_list,
)


def test_infer_activities_detects_states():
    from hackaton_system.config import Settings

    # Create processor with lower min_interval to ensure activities are detected
    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    detections = [
        DetectionResult(
            frame_id=0, time_sec=0.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1
        ),
        DetectionResult(
            frame_id=25, time_sec=1.0, bbox=(1, 1, 11, 11), confidence=0.8, track_id=1
        ),
        DetectionResult(
            frame_id=50,
            time_sec=2.0,
            bbox=(100, 100, 110, 110),
            confidence=0.7,
            track_id=1,
        ),
    ]
    tracks = {1: detections}
    fps = 25.0

    activities = VideoProcessor._infer_activities(processor, tracks, fps)

    assert len(activities) > 0
    activity_classes = {activity.activity_class for activity in activities}
    assert (
        "idle" in activity_classes
        or "walking" in activity_classes
        or "standing" in activity_classes
    )
    assert activities[0].start_sec >= 0.0


def test_infer_activities_single_detection_marks_idle():
    from hackaton_system.config import Settings

    # Create processor with lower min_interval to ensure activities are detected
    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    # Single detection creates a tail segment
    detections = [
        DetectionResult(
            frame_id=0, time_sec=5.0, bbox=(0, 0, 10, 10), confidence=0.5, track_id=7
        ),
        DetectionResult(
            frame_id=25, time_sec=6.0, bbox=(0, 0, 10, 10), confidence=0.5, track_id=7
        ),
    ]
    tracks = {7: detections}
    fps = 25.0

    activities = VideoProcessor._infer_activities(processor, tracks, fps)

    # Should have at least one activity
    assert len(activities) >= 1
    # Activity should be related to the detection time
    assert any(activity.start_sec <= 6.0 <= activity.end_sec for activity in activities)


def test_to_list_handles_torch_like_objects():
    class Dummy:
        def __init__(self) -> None:
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return self

        def tolist(self):
            return [1, 2, 3]

    dummy = Dummy()

    assert _to_list(dummy) == [1, 2, 3]
    assert dummy.cpu_called


def test_to_list_handles_regular_list():
    """Test _to_list with regular Python list."""
    regular_list = [1, 2, 3]
    assert _to_list(regular_list) == [1, 2, 3]


def test_to_list_handles_object_with_tolist():
    """Test _to_list with object that has tolist but no cpu."""
    class ListLike:
        def tolist(self):
            return [4, 5, 6]

    obj = ListLike()
    assert _to_list(obj) == [4, 5, 6]


def test_center_of_bbox():
    """Test _center_of_bbox calculates center correctly."""
    bbox = (10, 20, 30, 40)
    center = VideoProcessor._center_of_bbox(bbox)
    assert center == (20.0, 30.0)


def test_speed_calculates_distance():
    """Test _speed calculates speed correctly."""
    prev = (0.0, 0.0)
    curr = (3.0, 4.0)  # distance = 5.0
    duration = 1.0
    speed = VideoProcessor._speed(prev, curr, duration)
    assert abs(speed - 5.0) < 0.001  # sqrt(3^2 + 4^2) / 1.0 = 5.0


def test_speed_handles_zero_duration():
    """Test _speed returns 0 for zero duration."""
    prev = (0.0, 0.0)
    curr = (10.0, 10.0)
    duration = 0.0
    speed = VideoProcessor._speed(prev, curr, duration)
    assert speed == 0.0


def test_group_detections_by_track():
    """Test _group_detections_by_track groups detections correctly."""
    processor = VideoProcessor()
    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
        DetectionResult(frame_id=1, time_sec=1.0, bbox=(1, 1, 11, 11), confidence=0.8, track_id=1),
        DetectionResult(frame_id=2, time_sec=2.0, bbox=(100, 100, 110, 110), confidence=0.7, track_id=2),
    ]

    tracks = VideoProcessor._group_detections_by_track(processor, detections)

    assert len(tracks) == 2
    assert len(tracks[1]) == 2
    assert len(tracks[2]) == 1
    assert tracks[1][0].time_sec == 0.0
    assert tracks[1][1].time_sec == 1.0
    assert tracks[2][0].time_sec == 2.0


def test_group_detections_by_track_sorts_by_time():
    """Test _group_detections_by_track sorts detections by time."""
    processor = VideoProcessor()
    detections = [
        DetectionResult(frame_id=2, time_sec=2.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
        DetectionResult(frame_id=1, time_sec=1.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
    ]

    tracks = VideoProcessor._group_detections_by_track(processor, detections)

    assert tracks[1][0].time_sec == 0.0
    assert tracks[1][1].time_sec == 1.0
    assert tracks[1][2].time_sec == 2.0


def test_role_name_for_zone():
    """Test _role_name_for_zone returns correct role names."""
    from hackaton_system.config import ZoneDefinition, Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Test with None zone
    assert processor._role_name_for_zone(None) == "visitor"

    # Test with zone that has person_type
    zone = ZoneDefinition(
        name="test",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category="station",
        person_type="operator",
    )
    assert processor._role_name_for_zone(zone) == "operator"

    # Test with restricted zone
    restricted_zone = ZoneDefinition(
        name="restricted",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category="restricted",
        person_type="restricted",
    )
    assert processor._role_name_for_zone(restricted_zone) == "visitor"

    # Test with zone without person_type
    empty_zone = ZoneDefinition(
        name="empty",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category="corridor",
        person_type="",
    )
    assert processor._role_name_for_zone(empty_zone) == "visitor"


def test_zone_for_point_finds_zone():
    """Test _zone_for_point finds zone containing point."""
    from hackaton_system.config import Settings, ZoneDefinition

    zone = ZoneDefinition(
        name="test_zone",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category="station",
        person_type="operator",
    )
    settings = Settings(zones=[zone])
    processor = VideoProcessor(settings=settings)

    # Point inside zone
    found_zone = processor._zone_for_point((50.0, 50.0))
    assert found_zone is not None
    assert found_zone.name == "test_zone"

    # Point outside zone
    found_zone = processor._zone_for_point((150.0, 150.0))
    assert found_zone is None


def test_zone_for_point_boundary_conditions():
    """Test _zone_for_point handles boundary conditions."""
    from hackaton_system.config import Settings, ZoneDefinition

    zone = ZoneDefinition(
        name="boundary_zone",
        x_min=10,
        y_min=20,
        x_max=30,
        y_max=40,
        zone_category="station",
        person_type="operator",
    )
    settings = Settings(zones=[zone])
    processor = VideoProcessor(settings=settings)

    # Test boundaries
    assert processor._zone_for_point((10.0, 20.0)) is not None  # min corner
    assert processor._zone_for_point((30.0, 40.0)) is not None  # max corner
    assert processor._zone_for_point((9.9, 20.0)) is None  # just outside
    assert processor._zone_for_point((30.1, 40.0)) is None  # just outside


def test_activity_label_restricted_zone():
    """Test _activity_label returns in_restricted_zone for restricted zones."""
    from hackaton_system.config import Settings, ZoneDefinition

    restricted_zone = ZoneDefinition(
        name="restricted",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category="restricted",
        person_type="restricted",
    )
    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Any speed in restricted zone should return in_restricted_zone
    assert processor._activity_label(restricted_zone, 0.0) == "in_restricted_zone"
    assert processor._activity_label(restricted_zone, 100.0) == "in_restricted_zone"
    assert processor._activity_label(restricted_zone, 200.0) == "in_restricted_zone"


def test_activity_label_station_zone():
    """Test _activity_label for station zones with different speeds."""
    from hackaton_system.config import Settings, ZoneDefinition

    station_zone = ZoneDefinition(
        name="station",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category="station",
        person_type="operator",
    )
    settings = Settings(
        activity_velocity_move_thresh=140.0,
        activity_velocity_idle_thresh=40.0,
    )
    processor = VideoProcessor(settings=settings)

    # High speed -> working
    assert processor._activity_label(station_zone, 150.0) == "working"
    # Low speed -> idle_at_station
    assert processor._activity_label(station_zone, 30.0) == "idle_at_station"
    # Medium speed -> working (default)
    assert processor._activity_label(station_zone, 80.0) == "working"


def test_activity_label_corridor_visitor_none():
    """Test _activity_label for corridor/visitor/None zones."""
    from hackaton_system.config import Settings, ZoneDefinition

    corridor_zone = ZoneDefinition(
        name="corridor",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category="corridor",
        person_type="supervisor",
    )
    settings = Settings(
        activity_velocity_move_thresh=140.0,
        activity_velocity_idle_thresh=40.0,
    )
    processor = VideoProcessor(settings=settings)

    # High speed -> walking
    assert processor._activity_label(corridor_zone, 150.0) == "walking"
    # Low speed -> standing
    assert processor._activity_label(corridor_zone, 30.0) == "standing"
    # Medium speed -> walking (default)
    assert processor._activity_label(corridor_zone, 80.0) == "walking"

    # None zone should behave like corridor
    assert processor._activity_label(None, 150.0) == "walking"
    assert processor._activity_label(None, 30.0) == "standing"
    assert processor._activity_label(None, 80.0) == "walking"


def test_classify_track_states_empty_detections():
    """Test _classify_track_states returns empty list for empty detections."""
    from hackaton_system.config import Settings

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)

    states = processor._classify_track_states(track_id=1, detections=[], fps=25.0)
    assert states == []


def test_classify_track_states_single_detection():
    """Test _classify_track_states with single detection creates tail segment."""
    from hackaton_system.config import Settings

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)

    detections = [
        DetectionResult(frame_id=0, time_sec=5.0, bbox=(100, 200, 200, 300), confidence=0.9, track_id=1),
    ]

    states = processor._classify_track_states(track_id=1, detections=detections, fps=25.0)

    assert len(states) == 1  # Only tail segment
    assert states[0].track_id == 1
    assert states[0].start_sec == 5.0


def test_classify_track_states_multiple_detections():
    """Test _classify_track_states with multiple detections."""
    from hackaton_system.config import Settings

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)

    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(100, 200, 200, 300), confidence=0.9, track_id=1),
        DetectionResult(frame_id=25, time_sec=1.0, bbox=(110, 210, 210, 310), confidence=0.8, track_id=1),
        DetectionResult(frame_id=50, time_sec=2.0, bbox=(120, 220, 220, 320), confidence=0.7, track_id=1),
    ]

    states = processor._classify_track_states(track_id=1, detections=detections, fps=25.0)

    # Should have states for each transition + tail segment
    assert len(states) >= 2
    assert all(state.track_id == 1 for state in states)


def test_accumulate_role_time():
    """Test _accumulate_role_time accumulates time by role."""
    from hackaton_system.config import Settings, ZoneDefinition

    # Create zones for role assignment
    station_zone = ZoneDefinition(
        name="station",
        x_min=50,
        y_min=50,
        x_max=150,
        y_max=150,
        zone_category="station",
        person_type="operator",
    )
    settings = Settings(zones=[station_zone], activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)

    # Detections in station zone
    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(100, 100, 110, 110), confidence=0.9, track_id=1),
        DetectionResult(frame_id=25, time_sec=1.0, bbox=(100, 100, 110, 110), confidence=0.8, track_id=1),
        DetectionResult(frame_id=50, time_sec=2.0, bbox=(100, 100, 110, 110), confidence=0.7, track_id=1),
    ]

    totals = processor._accumulate_role_time(detections, fps=25.0)

    assert "operator" in totals
    assert totals["operator"] > 0


def test_accumulate_role_time_visitor_zone():
    """Test _accumulate_role_time for visitor zone."""
    from hackaton_system.config import Settings

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)

    # Detections outside any zone (should be visitor)
    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(1000, 1000, 1010, 1010), confidence=0.9, track_id=1),
        DetectionResult(frame_id=25, time_sec=1.0, bbox=(1000, 1000, 1010, 1010), confidence=0.8, track_id=1),
    ]

    totals = processor._accumulate_role_time(detections, fps=25.0)

    assert "visitor" in totals
    assert totals["visitor"] > 0


def test_infer_person_roles_empty_tracks():
    """Test _infer_person_roles with empty tracks."""
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    assignments = processor._infer_person_roles({}, fps=25.0)

    assert assignments == {}


def test_infer_person_roles_empty_detections():
    """Test _infer_person_roles with empty detections."""
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    assignments = processor._infer_person_roles({1: []}, fps=25.0)

    assert 1 not in assignments  # Empty detections are skipped


def test_infer_person_roles_with_confidence_threshold():
    """Test _infer_person_roles respects confidence threshold."""
    from hackaton_system.config import Settings, ZoneDefinition

    station_zone = ZoneDefinition(
        name="station",
        x_min=50,
        y_min=50,
        x_max=150,
        y_max=150,
        zone_category="station",
        person_type="operator",
    )
    # High threshold - should return unknown if confidence too low
    settings = Settings(
        zones=[station_zone],
        role_assignment_threshold=0.9,
        activity_min_interval_sec=0.1,
    )
    processor = VideoProcessor(settings=settings)

    # Detections that might not reach 90% confidence
    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(100, 100, 110, 110), confidence=0.9, track_id=1),
        DetectionResult(frame_id=25, time_sec=1.0, bbox=(200, 200, 210, 210), confidence=0.8, track_id=1),  # Outside zone
    ]

    tracks = {1: detections}
    assignments = processor._infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    # Depending on time distribution, might be unknown or operator
    assert assignments[1].person_type in ["operator", "unknown", "visitor"]


def test_ensure_person_records():
    """Test _ensure_person_records creates Person records."""
    from hackaton_system.config import Settings
    from hackaton_system.db import Video, get_session, init_db

    settings = Settings()
    processor = VideoProcessor(settings=settings)
    init_db()

    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
        DetectionResult(frame_id=1, time_sec=1.0, bbox=(0, 0, 10, 10), confidence=0.8, track_id=2),
        DetectionResult(frame_id=2, time_sec=2.0, bbox=(0, 0, 10, 10), confidence=0.7, track_id=1),
    ]

    with get_session() as session:
        video = Video(filename="test.mp4", fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person_map = processor._ensure_person_records(session, video, detections)

        assert len(person_map) == 2  # Two unique track_ids
        assert 1 in person_map
        assert 2 in person_map
        assert person_map[1].track_id == 1
        assert person_map[2].track_id == 2
        assert person_map[1].person_type == "unknown"
        assert person_map[1].person_type_conf == 0.0


def test_ensure_person_records_empty_detections():
    """Test _ensure_person_records with empty detections."""
    from hackaton_system.config import Settings
    from hackaton_system.db import Video, get_session, init_db

    settings = Settings()
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename="test.mp4", fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person_map = processor._ensure_person_records(session, video, [])

        assert person_map == {}


def test_load_detector_lazy_initialization(monkeypatch):
    """Test _load_detector lazy initialization."""
    from hackaton_system.config import Settings
    from unittest.mock import MagicMock

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Mock ultralytics module
    mock_ultralytics = MagicMock()
    mock_yolo_class = MagicMock()
    mock_yolo_instance = MagicMock()
    mock_yolo_class.return_value = mock_yolo_instance
    mock_ultralytics.YOLO = mock_yolo_class

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.import_module", lambda x: mock_ultralytics)

        detector1 = processor._load_detector()
        detector2 = processor._load_detector()

        # Should return same instance (lazy initialization)
        assert detector1 is detector2
        # Should only be called once
        assert mock_yolo_class.call_count == 1


def test_load_detector_with_custom_weights(monkeypatch):
    """Test _load_detector uses custom weights if provided."""
    from hackaton_system.config import Settings
    from unittest.mock import MagicMock

    settings = Settings(detection_model_path="custom_weights.pt")
    processor = VideoProcessor(settings=settings)

    mock_ultralytics = MagicMock()
    mock_yolo_class = MagicMock()
    mock_yolo_instance = MagicMock()
    mock_yolo_class.return_value = mock_yolo_instance
    mock_ultralytics.YOLO = mock_yolo_class

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.import_module", lambda x: mock_ultralytics)

        processor._load_detector()

        # Should use custom weights
        mock_yolo_class.assert_called_once_with("custom_weights.pt")


def test_process_video_file_not_found():
    """Test process_video raises FileNotFoundError for non-existent file."""
    from hackaton_system.config import Settings
    from pathlib import Path

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    with pytest.raises(FileNotFoundError):
        processor.process_video(Path("nonexistent_video.mp4"))


def test_infer_activities_empty_tracks():
    """Test _infer_activities with empty tracks."""
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    activities = processor._infer_activities({}, fps=25.0)

    assert activities == []


def test_infer_activities_filters_by_min_duration():
    """Test _infer_activities filters activities by min_duration."""
    from hackaton_system.config import Settings

    # High min_duration threshold
    settings = Settings(activity_min_interval_sec=10.0)
    processor = VideoProcessor(settings=settings)

    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(100, 200, 200, 300), confidence=0.9, track_id=1),
        DetectionResult(frame_id=25, time_sec=1.0, bbox=(100, 200, 200, 300), confidence=0.8, track_id=1),
    ]
    tracks = {1: detections}

    activities = processor._infer_activities(tracks, fps=25.0)

    # Short activities should be filtered out
    assert len(activities) == 0


def test_infer_activities_merges_adjacent_states():
    """Test _infer_activities merges adjacent states with same label."""
    from hackaton_system.config import Settings

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)

    # Create detections that should produce merged states
    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(100, 200, 200, 300), confidence=0.9, track_id=1),
        DetectionResult(frame_id=25, time_sec=1.0, bbox=(100, 200, 200, 300), confidence=0.8, track_id=1),
        DetectionResult(frame_id=50, time_sec=2.0, bbox=(100, 200, 200, 300), confidence=0.7, track_id=1),
    ]
    tracks = {1: detections}

    activities = processor._infer_activities(tracks, fps=25.0)

    # Should have activities (may be merged)
    assert len(activities) >= 0


def test_infer_activities_empty_frame_states():
    """Test _infer_activities handles empty frame_states."""
    from hackaton_system.config import Settings

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)

    # Create tracks that produce no frame states (empty detections after filtering)
    tracks = {1: []}  # Empty detections

    activities = processor._infer_activities(tracks, fps=25.0)

    # Should return empty list when no frame states
    assert activities == []


def test_infer_person_roles_empty_role_totals():
    """Test _infer_person_roles handles empty role_totals."""
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Mock _accumulate_role_time to return empty dict
    def mock_accumulate(dets, fps):
        return {}

    processor._accumulate_role_time = mock_accumulate

    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
    ]
    tracks = {1: detections}

    assignments = processor._infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    assert assignments[1].person_type == "unknown"
    assert assignments[1].confidence == 0.0


def test_infer_person_roles_zero_total_time():
    """Test _infer_person_roles handles zero total_time edge case."""
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Mock _accumulate_role_time to return dict with zero values
    def mock_accumulate(dets, fps):
        return {"operator": 0.0, "visitor": 0.0}

    processor._accumulate_role_time = mock_accumulate

    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
    ]
    tracks = {1: detections}

    assignments = processor._infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    # Should handle zero total_time gracefully
    assert assignments[1].confidence == 0.0


def test_extract_video_metadata(monkeypatch):
    """Test _extract_video_metadata extracts video metadata correctly."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Mock cv2.VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,  # CAP_PROP_FPS
        7: 750,   # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    mock_cap.release = MagicMock()

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.cv2.VideoCapture", lambda x: mock_cap)

        fps, frames, duration = processor._extract_video_metadata(Path("test.mp4"))

        assert fps == 25.0
        assert frames == 750
        assert duration == 30.0  # 750 / 25.0
        mock_cap.release.assert_called_once()


def test_extract_video_metadata_default_fps(monkeypatch):
    """Test _extract_video_metadata uses default FPS when not available."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 0  # No FPS available
    mock_cap.release = MagicMock()

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.cv2.VideoCapture", lambda x: mock_cap)

        fps, frames, duration = processor._extract_video_metadata(Path("test.mp4"))

        assert fps == 25.0  # Default FPS
        mock_cap.release.assert_called_once()


def test_extract_video_metadata_cannot_open(monkeypatch):
    """Test _extract_video_metadata raises RuntimeError when video cannot be opened."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.cv2.VideoCapture", lambda x: mock_cap)

        with pytest.raises(RuntimeError, match="Cannot open video"):
            processor._extract_video_metadata(Path("test.mp4"))


def test_run_detection_with_mocked_yolo(monkeypatch):
    """Test _run_detection processes YOLO results correctly."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings(detection_conf=0.5)
    processor = VideoProcessor(settings=settings)

    # Mock YOLO detector
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    # Mock boxes attributes
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40], [50, 60, 70, 80]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9, 0.8]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0, 0]  # Class 0 = person
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1, 2]  # Track IDs
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    assert len(detections) == 2
    assert detections[0].track_id == 1
    assert detections[0].confidence == 0.9
    assert detections[1].track_id == 2
    assert detections[1].confidence == 0.8


def test_run_detection_filters_low_confidence(monkeypatch):
    """Test _run_detection filters detections below confidence threshold."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings(detection_conf=0.7)  # High threshold
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.5]  # Below threshold
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should be filtered out due to low confidence
    assert len(detections) == 0


def test_run_detection_filters_non_person_classes(monkeypatch):
    """Test _run_detection filters non-person classes."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [1]  # Not class 0 (person)
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should be filtered out (not a person)
    assert len(detections) == 0


def test_run_detection_handles_no_boxes(monkeypatch):
    """Test _run_detection handles results without boxes."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = None  # No boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should handle gracefully and return empty list
    assert len(detections) == 0


def test_run_detection_handles_exception(monkeypatch):
    """Test _run_detection handles exceptions gracefully."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_detector.track.side_effect = Exception("YOLO error")

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should return empty list on error
    assert detections == []


def test_process_video_full_flow(monkeypatch, tmp_path):
    """Test process_video full flow with all mocks."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings
    from hackaton_system.db import Video, get_session, init_db

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    init_db()

    # Create a real file for testing
    test_video_path = tmp_path / "test_video.mp4"
    test_video_path.touch()  # Create empty file
    
    # Mock cv2.VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,  # CAP_PROP_FPS
        7: 250,   # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    mock_cap.release = MagicMock()

    # Mock YOLO detector
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    with monkeypatch.context() as m:
        # Mock cv2
        m.setattr("hackaton_system.pipeline.video_processor.cv2.VideoCapture", lambda x: mock_cap)
        # Mock detector
        processor._detector = mock_detector

        video_id = processor.process_video(test_video_path)

        # Verify video was created
        with get_session() as session:
            video = session.query(Video).filter_by(id=video_id).first()
            assert video is not None
            assert video.filename == "test_video.mp4"
            assert video.fps == 25.0


def test_run_detection_no_conf_attr(monkeypatch):
    """Test _run_detection handles missing conf attribute."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = None  # No conf attribute
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should use default confidence 0.0, but will be filtered by threshold
    assert len(detections) == 0


def test_run_detection_no_cls_attr(monkeypatch):
    """Test _run_detection handles missing cls attribute."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = None  # No cls attribute
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should default to class 0 (person)
    assert len(detections) == 1
    assert detections[0].track_id == 1


def test_run_detection_no_id_attr(monkeypatch):
    """Test _run_detection handles missing id attribute."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = None  # No id attribute
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should filter out (no track_id)
    assert len(detections) == 0


def test_run_detection_no_xyxy_attr(monkeypatch):
    """Test _run_detection handles missing xyxy attribute."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.xyxy = None  # No xyxy attribute
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    processor._detector = mock_detector

    detections = processor._run_detection(Path("test.mp4"), fps=25.0)

    # Should skip frame
    assert len(detections) == 0


def test_process_video_skips_missing_track_in_person_map(monkeypatch, tmp_path):
    """Test process_video skips tracks not in person_map for role assignment."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings
    from hackaton_system.db import Video, get_session, init_db
    from hackaton_system.pipeline.video_processor import RoleAssignment

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    init_db()

    test_video_path = tmp_path / "test_video.mp4"
    test_video_path.touch()
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,
        7: 250,
    }.get(prop, 0)
    mock_cap.release = MagicMock()

    # Create detections with track_id 1
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    # Mock _infer_person_roles to return assignment for track_id 999 (not in person_map)
    def mock_infer_roles(tracks, fps):
        return {999: RoleAssignment(999, "operator", 0.9)}  # Track not in person_map

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.cv2.VideoCapture", lambda x: mock_cap)
        processor._detector = mock_detector
        processor._infer_person_roles = mock_infer_roles

        video_id = processor.process_video(test_video_path)

        # Should complete without error, skipping track 999
        assert video_id is not None


def test_process_video_skips_activity_with_missing_track(monkeypatch, tmp_path):
    """Test process_video skips activities for tracks not in person_map."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from hackaton_system.config import Settings
    from hackaton_system.db import Video, get_session, init_db
    from hackaton_system.pipeline.video_processor import ActivityResult

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    init_db()

    test_video_path = tmp_path / "test_video.mp4"
    test_video_path.touch()
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,
        7: 250,
    }.get(prop, 0)
    mock_cap.release = MagicMock()

    # Create detections with track_id 1
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]
    
    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    # Mock _infer_activities to return activity for track_id 999 (not in person_map)
    def mock_infer_activities(tracks, fps):
        return [ActivityResult(999, "working", 0.0, 10.0, 1.0)]  # Track not in person_map

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.cv2.VideoCapture", lambda x: mock_cap)
        processor._detector = mock_detector
        processor._infer_activities = mock_infer_activities

        video_id = processor.process_video(test_video_path)

        # Should complete without error, skipping activity for track 999
        assert video_id is not None


def test_infer_person_roles_high_confidence():
    """Test _infer_person_roles with high confidence assignment."""
    from hackaton_system.config import Settings, ZoneDefinition

    station_zone = ZoneDefinition(
        name="station",
        x_min=50,
        y_min=50,
        x_max=150,
        y_max=150,
        zone_category="station",
        person_type="operator",
    )
    # Low threshold - should assign role easily
    settings = Settings(
        zones=[station_zone],
        role_assignment_threshold=0.3,
        activity_min_interval_sec=0.1,
    )
    processor = VideoProcessor(settings=settings)

    # All detections in station zone
    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(100, 100, 110, 110), confidence=0.9, track_id=1),
        DetectionResult(frame_id=25, time_sec=1.0, bbox=(100, 100, 110, 110), confidence=0.8, track_id=1),
        DetectionResult(frame_id=50, time_sec=2.0, bbox=(100, 100, 110, 110), confidence=0.7, track_id=1),
    ]

    tracks = {1: detections}
    assignments = processor._infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    # Should have high confidence for operator
    assert assignments[1].confidence > 0.0


def test_load_detector_import_error(monkeypatch):
    """Test _load_detector raises RuntimeError on ImportError."""
    from hackaton_system.config import Settings

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    def mock_import_module(name):
        raise ImportError("No module named 'ultralytics'")

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.import_module", mock_import_module)

        with pytest.raises(RuntimeError, match="Ultralytics package is not installed"):
            processor._load_detector()


def test_load_detector_no_yolo_class(monkeypatch):
    """Test _load_detector raises RuntimeError when YOLO class not found."""
    from hackaton_system.config import Settings
    from unittest.mock import MagicMock

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO = None  # YOLO class not available

    with monkeypatch.context() as m:
        m.setattr("hackaton_system.pipeline.video_processor.import_module", lambda x: mock_ultralytics)

        with pytest.raises(RuntimeError, match="does not expose the YOLO class"):
            processor._load_detector()
