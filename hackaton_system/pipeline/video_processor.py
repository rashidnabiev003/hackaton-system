from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, List, Protocol, Sequence, cast

import cv2
from sqlalchemy.orm import Session

from hackaton_system.config import Settings, get_settings
from hackaton_system.db import (
    Activity,
    Detection,
    Person,
    PoseKeypoints,
    Video,
    get_session,
    init_db,
)

LOGGER = logging.getLogger(__name__)


class DetectorProtocol(Protocol):
    """Minimal interface required from the Ultralytics YOLO model."""

    def track(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        ...


def _to_list(obj: Any) -> List[Any]:
    """Return a best-effort Python list from torch/np/list inputs."""

    array = obj.cpu() if hasattr(obj, "cpu") else obj
    if hasattr(array, "tolist"):
        array = array.tolist()
    return list(array)


@dataclass(slots=True)
class DetectionResult:
    frame_id: int
    time_sec: float
    bbox: tuple[int, int, int, int]
    confidence: float
    track_id: int


@dataclass(slots=True)
class PoseResult:
    frame_id: int
    time_sec: float
    track_id: int
    keypoints: list[dict[str, float]]
    confidence: float


@dataclass(slots=True)
class ActivityResult:
    track_id: int
    activity_class: str
    start_sec: float
    end_sec: float
    confidence: float


class VideoProcessor:
    """High level orchestrator: detection → tracking → DB writer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._detector: DetectorProtocol | None = None
        # При создании процессора убеждаемся, что схема БД готова принимать данные.
        init_db()

    def process_video(self, video_path: str | Path) -> int:
        """Process a single video file and persist metadata.

        Returns the created video_id.
        """

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        fps, frame_count, duration = self._extract_video_metadata(video_path)
        LOGGER.info(
            "Processing %s (fps=%.2f, frames=%s, duration=%.2fs)",
            video_path.name,
            fps,
            frame_count,
            duration,
        )

        detections, poses = self._run_detection(video_path, fps=fps)
        activities = self._infer_activities(detections)

        with get_session() as session:
            # Фиксируем сам видеоролик и получаем первичный ключ (video.id).
            video = Video(filename=video_path.name, fps=fps, duration_sec=duration)
            session.add(video)
            session.flush()

            person_map = self._ensure_person_records(session=session, video=video, detections=detections)

            # Сохраняем покадровые детекции.
            for det in detections:
                bbox = det.bbox
                detection_row = Detection(
                    video_id=video.id,
                    person_id=person_map[det.track_id].id,
                    frame_id=det.frame_id,
                    time_sec=det.time_sec,
                    x_min=bbox[0],
                    y_min=bbox[1],
                    x_max=bbox[2],
                    y_max=bbox[3],
                    confidence=det.confidence,
                )
                session.add(detection_row)

            # Добавляем интервалы активностей, если соответствующий трек найден.
            for activity in activities:
                if activity.track_id not in person_map:
                    continue
                session.add(
                    Activity(
                        video_id=video.id,
                        person_id=person_map[activity.track_id].id,
                        activity_class=activity.activity_class,
                        t_start_sec=activity.start_sec,
                        t_end_sec=activity.end_sec,
                        activity_conf=activity.confidence,
                    )
                )

            if poses and self.settings.enable_pose_capture:
                for pose in poses:
                    if pose.track_id not in person_map:
                        continue
                    session.add(
                        PoseKeypoints(
                            video_id=video.id,
                            person_id=person_map[pose.track_id].id,
                            frame_id=pose.frame_id,
                            time_sec=pose.time_sec,
                            keypoints=pose.keypoints,
                            pose_conf=pose.confidence,
                        )
                    )

            session.flush()
            LOGGER.info(
                "Committed video_id=%s with %s detections and %s pose rows",
                video.id,
                len(detections),
                len(poses) if self.settings.enable_pose_capture else 0,
            )
            return video.id

    def _extract_video_metadata(self, video_path: Path) -> tuple[float, int, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if fps else 0.0
        cap.release()
        return fps, frames, duration

    def _run_detection(self, video_path: Path, fps: float) -> tuple[list[DetectionResult], list[PoseResult]]:
        """Выполняем детекцию + трекинг через Ultralytics YOLO."""

        model: DetectorProtocol = self._load_detector()
        detections: list[DetectionResult] = []
        poses: list[PoseResult] = []
        frame_idx = 0

        try:
            results_stream: Iterable[Any] = model.track(
                source=str(video_path),
                tracker="bytetrack.yaml",
                stream=True,
                save=False,
                show=False,
                verbose=False,
            )
            for result in results_stream:
                boxes: Any = getattr(result, "boxes", None)
                xyxy_attr = getattr(boxes, "xyxy", None) if boxes is not None else None
                if boxes is None or xyxy_attr is None:
                    frame_idx += 1
                    continue

                xyxy_source = xyxy_attr.cpu() if hasattr(xyxy_attr, "cpu") else xyxy_attr
                xyxy_data = xyxy_source.tolist() if hasattr(xyxy_source, "tolist") else xyxy_source
                xyxy: List[List[float]] = cast(List[List[float]], xyxy_data)

                conf_attr = getattr(boxes, "conf", None)
                confs: List[float]
                if conf_attr is not None:
                    confs = cast(List[float], _to_list(conf_attr))
                else:
                    confs = [0.0] * len(xyxy)

                cls_attr = getattr(boxes, "cls", None)
                classes: List[int | None]
                if cls_attr is not None:
                    classes = cast(List[int | None], _to_list(cls_attr))
                else:
                    classes = [0 for _ in xyxy]

                id_attr = getattr(boxes, "id", None)
                track_ids: List[int | None]
                if id_attr is not None:
                    track_ids = cast(List[int | None], _to_list(id_attr))
                else:
                    track_ids = [None for _ in xyxy]

                keypoints_attr = getattr(result, "keypoints", None)
                kp_xy_attr = getattr(keypoints_attr, "xy", None) if keypoints_attr is not None else None
                kp_conf_attr = getattr(keypoints_attr, "conf", None) if keypoints_attr is not None else None

                keypoints_xy: List[List[List[float]] | None]
                if kp_xy_attr is not None:
                    kp_xy_source = kp_xy_attr.cpu() if hasattr(kp_xy_attr, "cpu") else kp_xy_attr
                    kp_xy_data = kp_xy_source.tolist() if hasattr(kp_xy_source, "tolist") else kp_xy_source
                    keypoints_xy = cast(List[List[List[float]]], kp_xy_data)
                    if len(keypoints_xy) != len(xyxy):
                        keypoints_xy = [None for _ in xyxy]
                else:
                    keypoints_xy = [None for _ in xyxy]

                keypoints_conf: List[List[float] | None]
                if kp_conf_attr is not None:
                    kp_conf_source = kp_conf_attr.cpu() if hasattr(kp_conf_attr, "cpu") else kp_conf_attr
                    kp_conf_data = kp_conf_source.tolist() if hasattr(kp_conf_source, "tolist") else kp_conf_source
                    keypoints_conf = cast(List[List[float]], kp_conf_data)
                    if len(keypoints_conf) != len(xyxy):
                        keypoints_conf = [None for _ in xyxy]
                else:
                    keypoints_conf = [None for _ in xyxy]

                frame_time = frame_idx / fps if fps else frame_idx

                for bbox, conf, cls_id, track_id, kp_xy, kp_conf in zip(
                    xyxy, confs, classes, track_ids, keypoints_xy, keypoints_conf
                ):
                    if cls_id is not None and int(cls_id) != 0:
                        # сохраняем только людей (class 0 в COCO)
                        continue
                    if track_id is None:
                        continue
                    bbox_tuple = cast(
                        tuple[int, int, int, int],
                        tuple(int(coord) for coord in bbox),
                    )
                    detections.append(
                        DetectionResult(
                            frame_id=frame_idx,
                            time_sec=frame_time,
                            bbox=bbox_tuple,
                            confidence=float(conf),
                            track_id=int(track_id),
                        )
                    )
                    if kp_xy is not None:
                        confidences = kp_conf or []
                        valid_scores = [score for score in confidences if score is not None]
                        avg_conf = (
                            float(sum(valid_scores) / len(valid_scores)) if valid_scores else float(conf)
                        )
                        padded_conf = confidences if confidences else [None] * len(kp_xy)
                        keypoints_payload: list[dict[str, float]] = []
                        for idx_point, point in enumerate(kp_xy):
                            entry: dict[str, float] = {"x": float(point[0]), "y": float(point[1])}
                            score = padded_conf[idx_point] if idx_point < len(padded_conf) else None
                            if score is not None:
                                entry["confidence"] = float(score)
                            keypoints_payload.append(entry)
                        poses.append(
                            PoseResult(
                                frame_id=frame_idx,
                                time_sec=frame_time,
                                track_id=int(track_id),
                                keypoints=keypoints_payload,
                                confidence=avg_conf,
                            )
                        )
                frame_idx += 1

        except Exception as exc:
            LOGGER.error("Detection failed: %s", exc, exc_info=True)
            return [], []

        LOGGER.info(
            "Detection complete: %s frames processed, %s tracks, %s pose entries",
            frame_idx,
            len(detections),
            len(poses),
        )
        return detections, poses

    def _infer_activities(self, detections: Sequence[DetectionResult]) -> list[ActivityResult]:
        """Простейшая эвристика активности: двигается / стоит."""

        if not detections:
            return []

        by_track: dict[int, list[DetectionResult]] = defaultdict(list)
        for det in detections:
            by_track[det.track_id].append(det)

        activities: list[ActivityResult] = []
        movement_threshold = 15  # px

        for track_id, dets in by_track.items():
            dets.sort(key=lambda d: d.time_sec)
            if len(dets) < 2:
                activities.append(
                    ActivityResult(
                        track_id=track_id,
                        activity_class="idle",
                        start_sec=dets[0].time_sec,
                        end_sec=dets[-1].time_sec,
                        confidence=0.5,
                    )
                )
                continue

            state = None
            state_start = dets[0].time_sec
            state_scores: list[float] = []

            def flush_state(end_time: float) -> None:
                if state is None:
                    return
                score = float(sum(state_scores) / len(state_scores)) if state_scores else 0.0
                norm_conf = min(score / 50.0, 1.0) if state == "moving" else max(1 - score / 30.0, 0.2)
                activities.append(
                    ActivityResult(
                        track_id=track_id,
                        activity_class=state,
                        start_sec=state_start,
                        end_sec=end_time,
                        confidence=norm_conf,
                    )
                )

            for prev, curr in zip(dets, dets[1:]):
                prev_center = ((prev.bbox[0] + prev.bbox[2]) / 2, (prev.bbox[1] + prev.bbox[3]) / 2)
                curr_center = ((curr.bbox[0] + curr.bbox[2]) / 2, (curr.bbox[1] + curr.bbox[3]) / 2)
                diff = ((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2) ** 0.5
                current_state = "moving" if diff > movement_threshold else "idle"

                if state is None:
                    state = current_state
                    state_start = prev.time_sec
                    state_scores = [diff]
                    continue

                if current_state == state:
                    state_scores.append(diff)
                else:
                    flush_state(prev.time_sec)
                    state = current_state
                    state_start = prev.time_sec
                    state_scores = [diff]

            flush_state(dets[-1].time_sec)

        return activities

    def _ensure_person_records(
        self,
        session: Session,
        video: Video,
        detections: Iterable[DetectionResult],
    ) -> dict[int, Person]:
        track_ids = sorted({det.track_id for det in detections})
        person_map: dict[int, Person] = {}
        for track_id in track_ids:
            person = Person(video_id=video.id, track_id=track_id, person_type="unknown")
            session.add(person)
            session.flush()
            person_map[track_id] = person
        return person_map

    def _load_detector(self) -> DetectorProtocol:
        """Ленивая инициализация YOLO, чтобы не грузить веса каждый раз."""

        if self._detector is None:
            weights = self.settings.detection_model_path or "yolov8n.pt"
            LOGGER.info("Loading YOLO weights: %s", weights)
            try:
                ultralytics = import_module("ultralytics")
            except ImportError as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError(
                    "Ultralytics package is not installed. Install `ultralytics` to run detection."
                ) from exc

            model_cls = getattr(ultralytics, "YOLO", None)
            if model_cls is None:
                raise RuntimeError("Installed Ultralytics package does not expose the YOLO class.")

            self._detector = cast(DetectorProtocol, model_cls(weights))
        return self._detector
