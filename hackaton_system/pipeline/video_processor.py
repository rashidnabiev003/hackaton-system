from __future__ import annotations

import logging
import re
from collections import defaultdict
import json
import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import subprocess
from typing import Any, Iterable, List, Mapping, Protocol, Sequence, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sqlalchemy.orm import Session
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

try:  # torchreid is optional; fall back to torchvision if missing
    from torchreid.utils import FeatureExtractor  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    FeatureExtractor = None

try:  # OCR is optional
    import easyocr
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None

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

    def track(self, *args: Any, **kwargs: Any) -> Iterable[Any]: ...


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
    embedding: list[float] | None = None


@dataclass(slots=True)
class PoseResult:
    frame_id: int
    time_sec: float
    track_id: int
    keypoints: list[dict[str, float]]
    confidence: float


@dataclass(slots=True)
class TrainDetectionResult:
    frame_id: int
    time_sec: float
    bbox: tuple[int, int, int, int]
    confidence: float
    track_id: int
    number: str | None = None
    number_confidence: float | None = None


@dataclass(slots=True)
class ActivityResult:
    track_id: int
    activity_class: str
    start_sec: float
    end_sec: float
    confidence: float


@dataclass(slots=True)
class RoleAssignment:
    track_id: int
    person_type: str
    confidence: float


@dataclass(slots=True)
class FrameState:
    track_id: int
    label: str
    start_sec: float
    end_sec: float


class VideoProcessor:
    """High level orchestrator: detection → tracking → DB writer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._detector: DetectorProtocol | None = None
        self._reid_model: Any | None = None
        self._reid_uses_torchreid = False
        self._reid_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._reid_transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._train_ocr_reader: Any | None = None
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

        detections, poses, train_events = self._run_detection(video_path, fps=fps)
        if getattr(self.settings, "reid_enabled", True):
            detections = self._stitch_tracks_with_reid(detections)
        tracks = self._group_detections_by_track(detections)
        track_embeddings = self._compute_track_embeddings(tracks)
        activities = self._infer_activities(tracks, fps=fps)
        role_map = self._infer_person_roles(tracks, fps=fps)
        preview_path: Path | None = None

        with get_session() as session:
            # Фиксируем сам видеоролик и получаем первичный ключ (video.id).
            video = Video(filename=video_path.name, fps=fps, duration_sec=duration)
            session.add(video)
            session.flush()

            person_map = self._ensure_person_records(
                session=session,
                video=video,
                detections=detections,
                track_embeddings=track_embeddings,
            )

            for track_id, assignment in role_map.items():
                if track_id not in person_map:
                    continue
                person = person_map[track_id]
                person.person_type = assignment.person_type
                person.person_type_conf = assignment.confidence

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
            should_render = bool(getattr(self.settings, "enable_video_render", True))
            if should_render:
                try:
                    preview_path = self._render_preview(
                        source=video_path,
                        video_id=video.id,
                        detections=detections,
                        train_detections=train_events,
                        fps=fps,
                    )
                    if preview_path:
                        LOGGER.info("Preview video stored at %s", preview_path)
                except Exception as exc:  # pragma: no cover - visualization is optional
                    LOGGER.warning("Failed to render preview video: %s", exc)
            if train_events:
                self._store_train_events(video_path, video.id, train_events)
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

    def _run_detection(
        self, video_path: Path, fps: float
    ) -> tuple[list[DetectionResult], list[PoseResult], list[TrainDetectionResult]]:
        """????????? ???????? + ??????? ????? Ultralytics YOLO."""

        model: DetectorProtocol = self._load_detector()
        detections: list[DetectionResult] = []
        poses: list[PoseResult] = []
        trains: list[TrainDetectionResult] = []
        enable_train = bool(getattr(self.settings, "enable_train_detection", False))
        track_classes = [0, 6] if enable_train else [0]
        tracker_conf = (
            min(self.settings.detection_conf, self.settings.train_detection_conf)
            if enable_train
            else self.settings.detection_conf
        )
        frame_idx = 0

        try:
            results_stream: Iterable[Any] = model.track(
                source=str(video_path),
                tracker=self.settings.tracker_config_path,
                stream=True,
                save=False,
                show=False,
                verbose=False,
                classes=track_classes,
                conf=tracker_conf,
                iou=self.settings.detection_iou,
                imgsz=self.settings.detection_imgsz,
            )
            for result in results_stream:
                boxes: Any = getattr(result, "boxes", None)
                xyxy_attr = getattr(boxes, "xyxy", None) if boxes is not None else None
                if boxes is None or xyxy_attr is None:
                    frame_idx += 1
                    continue

                xyxy_source = (
                    xyxy_attr.cpu() if hasattr(xyxy_attr, "cpu") else xyxy_attr
                )
                xyxy_data = (
                    xyxy_source.tolist()
                    if hasattr(xyxy_source, "tolist")
                    else xyxy_source
                )
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
                frame_img = getattr(result, "orig_img", None)

                for bbox, conf, cls_id, track_id, kp_xy, kp_conf in zip(
                    xyxy, confs, classes, track_ids, keypoints_xy, keypoints_conf
                ):
                    if cls_id is None:
                        continue
                    cls_int = int(cls_id)
                    bbox_tuple = cast(
                        tuple[int, int, int, int],
                        tuple(int(coord) for coord in bbox),
                    )
                    if cls_int == 0:
                        if track_id is None:
                            continue
                        if conf < self.settings.detection_conf:
                            continue
                        embedding_vec = None
                        if frame_img is not None:
                            try:
                                embedding_vec = self._extract_embedding(frame_img, bbox_tuple)
                            except Exception:
                                embedding_vec = None
                        detections.append(
                            DetectionResult(
                                frame_id=frame_idx,
                                time_sec=frame_time,
                                bbox=bbox_tuple,
                                confidence=float(conf),
                                track_id=int(track_id),
                                embedding=embedding_vec,
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
                    elif enable_train and cls_int == 6:
                        if conf < self.settings.train_detection_conf:
                            continue
                        number = None
                        number_conf = None
                        if frame_img is not None:
                            number, number_conf = self._extract_train_number(frame_img, bbox_tuple)
                        trains.append(
                            TrainDetectionResult(
                                frame_id=frame_idx,
                                time_sec=frame_time,
                                bbox=bbox_tuple,
                                confidence=float(conf),
                                track_id=int(track_id) if track_id is not None else -1,
                                number=number,
                                number_confidence=number_conf,
                            )
                        )
                frame_idx += 1

        except Exception as exc:
            LOGGER.error("Detection failed: %s", exc, exc_info=True)
            return [], [], []

        LOGGER.info(
            "Detection complete: %s frames processed, %s person tracks, %s pose entries, %s trains",
            frame_idx,
            len(detections),
            len(poses),
            len(trains),
        )
        return detections, poses, trains

    def _group_detections_by_track(
        self, detections: Sequence[DetectionResult]
    ) -> dict[int, list[DetectionResult]]:
        grouped: dict[int, list[DetectionResult]] = defaultdict(list)
        for det in detections:
            grouped[det.track_id].append(det)
        for dets in grouped.values():
            dets.sort(key=lambda d: d.time_sec)
        return grouped

    def _infer_activities(
        self, tracks: Mapping[int, list[DetectionResult]], fps: float | None = None
    ) -> list[ActivityResult]:
        """Эвристическое определение активностей под заводской сценарий."""

        if not tracks:
            return []

        effective_fps = fps if fps and fps > 0 else 25.0

        frame_states: list[FrameState] = []
        for track_id, dets in tracks.items():
            if not dets:
                continue
            frame_states.extend(self._classify_track_states(track_id, dets, effective_fps))

        if not frame_states:
            return []

        activities: list[ActivityResult] = []
        min_duration = self.settings.activity_min_interval_sec

        frame_states.sort(key=lambda s: (s.track_id, s.start_sec))
        merged: list[FrameState] = []
        for state in frame_states:
            if not merged:
                merged.append(state)
                continue
            last = merged[-1]
            if (
                state.track_id == last.track_id
                and state.label == last.label
                and state.start_sec <= last.end_sec + 1e-3
            ):
                last.end_sec = max(last.end_sec, state.end_sec)
            else:
                merged.append(state)

        for state in merged:
            duration = state.end_sec - state.start_sec
            if duration < min_duration:
                continue
            activities.append(
                ActivityResult(
                    track_id=state.track_id,
                    activity_class=state.label,
                    start_sec=state.start_sec,
                    end_sec=state.end_sec,
                    confidence=1.0,
                )
            )

        return activities

    def _classify_track_states(
        self, track_id: int, detections: Sequence[DetectionResult], fps: float
    ) -> list[FrameState]:
        if len(detections) == 0:
            return []

        frame_step = 1.0 / fps if fps else 1.0 / 25.0
        states: list[FrameState] = []
        prev = detections[0]
        prev_center = self._center_of_bbox(prev.bbox)

        for curr in detections[1:]:
            curr_center = self._center_of_bbox(curr.bbox)
            duration = max(curr.time_sec - prev.time_sec, frame_step)
            speed = self._speed(prev_center, curr_center, duration)
            zone = self._zone_for_point(prev_center)
            label = self._activity_label(zone, speed)
            states.append(
                FrameState(
                    track_id=track_id,
                    label=label,
                    start_sec=prev.time_sec,
                    end_sec=prev.time_sec + duration,
                )
            )
            prev = curr
            prev_center = curr_center

        # Tail segment to cover final frame
        tail_zone = self._zone_for_point(prev_center)
        states.append(
            FrameState(
                track_id=track_id,
                label=self._activity_label(tail_zone, 0.0),
                start_sec=prev.time_sec,
                end_sec=prev.time_sec + frame_step,
            )
        )
        return states

    def _zone_for_point(self, center: tuple[float, float]) -> ZoneDefinition | None:
        x, y = center
        for zone in self.settings.zones:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return zone
        return None

    def _activity_label(self, zone: ZoneDefinition | None, speed: float) -> str:
        move = self.settings.activity_velocity_move_thresh
        idle = self.settings.activity_velocity_idle_thresh
        category = zone.zone_category if zone else None

        if category == "restricted":
            return "in_restricted_zone"
        if category == "station":
            if speed >= move:
                return "working"
            if speed <= idle:
                return "idle_at_station"
            return "working"
        # corridor/visitor/None
        if speed >= move:
            return "walking"
        if speed <= idle:
            return "standing"
        return "walking"

    def _infer_person_roles(
        self, tracks: Mapping[int, list[DetectionResult]], fps: float
    ) -> dict[int, RoleAssignment]:
        assignments: dict[int, RoleAssignment] = {}
        for track_id, dets in tracks.items():
            if not dets:
                continue
            role_totals = self._accumulate_role_time(dets, fps)
            if not role_totals:
                assignments[track_id] = RoleAssignment(track_id, "unknown", 0.0)
                continue
            best_role, best_duration = max(
                role_totals.items(), key=lambda item: item[1]
            )
            total_time = sum(role_totals.values())
            confidence = best_duration / total_time if total_time else 0.0
            person_type = (
                best_role
                if confidence >= self.settings.role_assignment_threshold
                else "unknown"
            )
            assignments[track_id] = RoleAssignment(track_id, person_type, confidence)
        return assignments

    def _accumulate_role_time(
        self, detections: Sequence[DetectionResult], fps: float
    ) -> dict[str, float]:
        frame_step = 1.0 / fps if fps else 1.0 / 25.0
        totals: defaultdict[str, float] = defaultdict(float)
        prev = detections[0]
        prev_center = self._center_of_bbox(prev.bbox)

        for curr in detections[1:]:
            duration = max(curr.time_sec - prev.time_sec, frame_step)
            zone = self._zone_for_point(prev_center)
            role_name = self._role_name_for_zone(zone)
            totals[role_name] += duration
            prev = curr
            prev_center = self._center_of_bbox(curr.bbox)

        role_name = self._role_name_for_zone(self._zone_for_point(prev_center))
        totals[role_name] += frame_step
        return totals

    def _role_name_for_zone(self, zone: ZoneDefinition | None) -> str:
        if zone is None:
            return "visitor"
        if zone.person_type == "restricted":
            return "visitor"
        if not zone.person_type:
            return "visitor"
        return zone.person_type

    @staticmethod
    def _center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _speed(
        prev: tuple[float, float], curr: tuple[float, float], duration: float
    ) -> float:
        if duration <= 0.0:
            return 0.0
        return math.hypot(curr[0] - prev[0], curr[1] - prev[1]) / duration

    def _ensure_person_records(
        self,
        session: Session,
        video: Video,
        detections: Iterable[DetectionResult],
        track_embeddings: Mapping[int, str] | None = None,
    ) -> dict[int, Person]:
        track_ids = sorted({det.track_id for det in detections})
        person_map: dict[int, Person] = {}
        for track_id in track_ids:
            descriptor = track_embeddings.get(track_id) if track_embeddings else None
            person = Person(
                video_id=video.id,
                track_id=track_id,
                person_type="unknown",
                person_type_conf=0.0,
                reid_descriptor=descriptor,
            )
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
                raise RuntimeError(
                    "Installed Ultralytics package does not expose the YOLO class."
                )

            self._detector = cast(DetectorProtocol, model_cls(weights))
        return self._detector

    def _stitch_tracks_with_reid(
        self, detections: list[DetectionResult]
    ) -> list[DetectionResult]:
        if not detections or not any(det.embedding for det in detections):
            return detections

        tracks = self._group_detections_by_track(detections)
        summaries: list[dict[str, Any]] = []
        for track_id, dets in tracks.items():
            embeddings = [
                np.array(det.embedding, dtype=np.float32) for det in dets if det.embedding
            ]
            if not embeddings:
                continue
            mean_vec = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            if not norm:
                continue
            summaries.append(
                {
                    "track_id": track_id,
                    "start": dets[0].time_sec,
                    "end": dets[-1].time_sec,
                    "embedding": mean_vec / norm,
                }
            )

        summaries.sort(key=lambda item: item["start"])
        if not summaries:
            return detections

        gap_limit = getattr(self.settings, "reid_time_gap_sec", 2.5)
        similarity_thresh = getattr(self.settings, "reid_similarity_threshold", 0.6)

        mapping: dict[int, int] = {track_id: track_id for track_id in tracks.keys()}
        history: list[dict[str, Any]] = []
        for summary in summaries:
            assigned = mapping[summary["track_id"]]
            best_match_id = None
            best_similarity = similarity_thresh
            for candidate in history:
                if summary["start"] < candidate["end"]:
                    continue
                gap = summary["start"] - candidate["end"]
                if gap > gap_limit or gap < 0:
                    continue
                sim = float(np.dot(summary["embedding"], candidate["embedding"]))
                if sim >= best_similarity:
                    best_similarity = sim
                    best_match_id = candidate["assigned_id"]
            if best_match_id is not None:
                mapping[summary["track_id"]] = best_match_id
                summary["assigned_id"] = best_match_id
            else:
                summary["assigned_id"] = assigned
            history.append(summary)

        for det in detections:
            det.track_id = mapping.get(det.track_id, det.track_id)
        return detections

    def _compute_track_embeddings(
        self, tracks: Mapping[int, list[DetectionResult]]
    ) -> dict[int, str]:
        result: dict[int, str] = {}
        for track_id, dets in tracks.items():
            embeddings = [
                np.array(det.embedding, dtype=np.float32) for det in dets if det.embedding
            ]
            if not embeddings:
                continue
            mean_vec = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            if not norm:
                continue
            normalized = (mean_vec / norm).tolist()
            result[track_id] = json.dumps(normalized)
        return result

    def _render_preview(
        self,
        source: Path,
        video_id: int,
        detections: Sequence[DetectionResult],
        train_detections: Sequence[TrainDetectionResult] | None,
        fps: float,
    ) -> Path | None:
        """Собрать отдельное видео с разметкой боксов."""

        if not source.exists():
            return None

        output_dir = getattr(self.settings, "video_output_dir", Path("runs/visualizations"))
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_step_setting = max(1, int(getattr(self.settings, "preview_frame_step", 2)))
        max_side = max(1, int(getattr(self.settings, "preview_max_side", 640)))

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps_src = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
        if width <= 0 or height <= 0:
            cap.release()
            return None

        scale = 1.0
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        if target_width % 2:
            target_width += 1
        if target_height % 2:
            target_height += 1

        fps_out = max(fps_src / frame_step_setting, 1.0)

        ffmpeg_bin = getattr(self.settings, "preview_ffmpeg_path", None) or "ffmpeg"
        output_path = output_dir / f"{video_id}_{source.stem}.mp4"
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                LOGGER.warning("Could not remove existing preview: %s", output_path)
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{target_width}x{target_height}",
            "-r",
            f"{fps_out}",
            "-i",
            "-",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            str(output_path),
        ]

        try:
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        except FileNotFoundError as exc:  # pragma: no cover - depends on env
            LOGGER.warning("ffmpeg binary not found: %s", exc)
            return None

        frame_map: dict[int, list[DetectionResult]] = defaultdict(list)
        for det in detections:
            frame_map[det.frame_id].append(det)
        train_map: dict[int, list[TrainDetectionResult]] = defaultdict(list)
        for train in train_detections or []:
            train_map[train.frame_id].append(train)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step_setting != 0:
                    frame_idx += 1
                    continue
                if scale != 1.0:
                    annotated = cv2.resize(
                        frame, (target_width, target_height), interpolation=cv2.INTER_AREA
                    )
                else:
                    annotated = frame
                ratio_x = annotated.shape[1] / width
                ratio_y = annotated.shape[0] / height
                for det in frame_map.get(frame_idx, []):
                    color = self._color_for_track(det.track_id)
                    x1, y1, x2, y2 = det.bbox
                    x1 = int(x1 * ratio_x)
                    x2 = int(x2 * ratio_x)
                    y1 = int(y1 * ratio_y)
                    y2 = int(y2 * ratio_y)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {det.track_id}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                for train in train_map.get(frame_idx, []):
                    tx1, ty1, tx2, ty2 = train.bbox
                    tx1 = int(tx1 * ratio_x)
                    tx2 = int(tx2 * ratio_x)
                    ty1 = int(ty1 * ratio_y)
                    ty2 = int(ty2 * ratio_y)
                    cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 0, 255), 3)
                    train_label = f"Train {train.track_id}"
                    if train.number:
                        train_label += f" #{train.number}"
                    cv2.putText(
                        annotated,
                        train_label,
                        (tx1, max(ty1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                if proc.stdin:
                    proc.stdin.write(annotated.tobytes())
                frame_idx += 1
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

        return output_path if output_path.exists() else None

    @staticmethod
    def _color_for_track(track_id: int) -> tuple[int, int, int]:
        """Детерминированный цвет (BGR) для конкретного трека."""

        value = (track_id * 37) % 255
        return (value, (value * 2) % 255, (value * 3) % 255)

    def _ensure_reid_model(self) -> Any | None:
        if not getattr(self.settings, "reid_enabled", True):
            return None
        if self._reid_model is None:
            if FeatureExtractor is not None:
                try:
                    model_name = getattr(self.settings, "reid_model_name", "osnet_x1_0")
                    model_path = getattr(self.settings, "reid_model_path", None)
                    device = "cuda" if self._reid_device.type == "cuda" else "cpu"
                    self._reid_model = FeatureExtractor(
                        model_name=model_name,
                        model_path=model_path,
                        device=device,
                    )
                    self._reid_uses_torchreid = True
                    LOGGER.info("Torchreid FeatureExtractor loaded: %s on %s", model_name, device)
                    return self._reid_model
                except Exception as exc:  # pragma: no cover - optional dependency
                    LOGGER.warning(
                        "Failed to initialize Torchreid extractor (%s): %s. Falling back to torchvision.",
                        getattr(self.settings, "reid_model_name", "osnet_x1_0"),
                        exc,
                    )
                    self._reid_model = None
            try:
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to load torchvision ReID fallback: %s", exc)
                return None
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(self._reid_device)
            self._reid_model = model
            self._reid_uses_torchreid = False
        return self._reid_model

    def _extract_embedding(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> list[float] | None:
        model = self._ensure_reid_model()
        if model is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if self._reid_uses_torchreid and FeatureExtractor is not None:
            try:
                features = model([pil_img])
            except Exception as exc:  # pragma: no cover - runtime issue
                LOGGER.debug("Torchreid inference failed: %s", exc)
                return None
            if isinstance(features, torch.Tensor):
                vec = features[0]
            else:
                vec = torch.tensor(features[0])
            vec = F.normalize(vec, dim=0)
            return vec.cpu().tolist()

        tensor = self._reid_transform(pil_img).unsqueeze(0).to(self._reid_device)
        with torch.no_grad():
            emb = model(tensor)
            emb = F.normalize(emb, dim=1)
        return emb.squeeze(0).cpu().tolist()

    def _ensure_train_reader(self):
        if easyocr is None:
            return None
        if self._train_ocr_reader is None:
            try:
                langs = getattr(self.settings, "train_number_langs", ["en"])
                use_gpu = torch.cuda.is_available()
                self._train_ocr_reader = easyocr.Reader(langs, gpu=use_gpu)
            except Exception as exc:  # pragma: no cover - optional dependency
                LOGGER.warning("Failed to initialize EasyOCR reader: %s", exc)
                self._train_ocr_reader = None
        return self._train_ocr_reader

    def _extract_train_number(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> tuple[str | None, float | None]:
        reader = self._ensure_train_reader()
        if reader is None:
            return None, None
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None, None

        crop = frame[y1:y2, x1:x2]
        roi_height = max(5, int((y2 - y1) * 0.45))
        roi = crop[:roi_height, :]
        results = reader.readtext(roi)
        best_text = None
        best_conf = 0.0
        min_len = getattr(self.settings, "train_number_min_length", 3)
        for _, text, conf in results:
            cleaned = re.sub(r"[^0-9A-Z]", "", text.upper())
            if len(cleaned) < min_len:
                continue
            if conf > best_conf:
                best_conf = float(conf)
                best_text = cleaned
        return best_text, (best_conf if best_text else None)

    def _store_train_events(
        self, video_path: Path, video_id: int, detections: Sequence[TrainDetectionResult]
    ) -> None:
        output_dir = getattr(self.settings, "train_event_output_dir", Path("runs/train_events"))
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.warning("Failed to create train event dir %s: %s", output_dir, exc)
            return
        payload = [
            {
                "frame_id": det.frame_id,
                "time_sec": det.time_sec,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "track_id": det.track_id,
                "number": det.number,
                "number_confidence": det.number_confidence,
            }
            for det in detections
        ]
        out_path = output_dir / f"{video_id}_{video_path.stem}_trains.json"
        try:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            LOGGER.info("Stored %s train detections at %s", len(detections), out_path)
        except OSError as exc:
            LOGGER.warning("Failed to write train detections: %s", exc)
