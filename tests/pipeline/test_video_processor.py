# pyright: reportPrivateUsage=false
"""Tests for lightweight pieces of the video processing pipeline."""

from __future__ import annotations

from hackaton_system.pipeline.video_processor import (
    DetectionResult,
    VideoProcessor,
    _to_list,
)


def test_infer_activities_detects_states():
    processor = VideoProcessor.__new__(VideoProcessor)
    detections = [
        DetectionResult(frame_id=0, time_sec=0.0, bbox=(0, 0, 10, 10), confidence=0.9, track_id=1),
        DetectionResult(frame_id=1, time_sec=1.0, bbox=(1, 1, 11, 11), confidence=0.8, track_id=1),
        DetectionResult(frame_id=2, time_sec=2.0, bbox=(100, 100, 110, 110), confidence=0.7, track_id=1),
    ]

    activities = VideoProcessor._infer_activities(processor, detections)

    assert {activity.activity_class for activity in activities} == {"idle", "moving"}
    assert activities[0].start_sec == 0.0


def test_infer_activities_single_detection_marks_idle():
    processor = VideoProcessor.__new__(VideoProcessor)
    detections = [
        DetectionResult(frame_id=0, time_sec=5.0, bbox=(0, 0, 10, 10), confidence=0.5, track_id=7)
    ]

    activities = VideoProcessor._infer_activities(processor, detections)

    assert len(activities) == 1
    assert activities[0].activity_class == "idle"
    assert activities[0].start_sec == activities[0].end_sec == 5.0


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
