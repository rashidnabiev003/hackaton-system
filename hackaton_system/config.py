from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ZoneDefinition(BaseModel):
    """Rectangle describing a physical zone with semantic meaning."""

    name: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    zone_category: Literal["station", "corridor", "restricted", "visitor"]
    person_type: str


class Settings(BaseSettings):
    """Centralized project configuration loaded from environment variables."""

    # Используем модельную конфигурацию Pydantic, чтобы подтягивать значения из .env
    # c префиксом HACKATON_. Эта секция по сути описывает «откуда» приходят данные.
    model_config = SettingsConfigDict(
        env_prefix="HACKATON_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # URL подключения к БД; по умолчанию используем локальный SQLite в каталоге data.
    database_url: str = Field(
        # default="sqlite:///data/hackaton.db",
        default="sqlite:///C:/Users/SKade/Documents/VScode/Hackaton/BD/Hackaton_db.db",
        description="SQLAlchemy-compatible database URL.",
    )
    # Путь к каталогу с данными — через него создаём директории при инициализации БД.
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for processed artifacts and SQLite database.",
    )
    # Опциональные пути к весам: оставляем None, чтобы брать значения по умолчанию из Ultralytics.
    detection_model_path: str | None = Field(
        default="./yolo11n.pt", description="./yolo11n.pt"
    )
    enable_pose_capture: bool = Field(
        default=True,
        description="Persist pose keypoints when the underlying YOLO weights expose them.",
    )
    enable_video_render: bool = Field(
        default=True,
        description="Render annotated video previews for processed runs.",
    )
    preview_max_side: int = Field(
        default=1920,
        description="Max width/height for rendered previews (frames are downscaled preserving aspect ratio).",
    )
    preview_frame_step: int = Field(
        default=1,
        description="Write every N-th frame to preview to reduce size (1 = every frame).",
    )
    preview_ffmpeg_path: str | None = Field(
        default="C:/Users/SKade/AppData/Roaming/ffmpeg/bin/ffmpeg.exe",
        description="Path to ffmpeg binary (if None, assumes ffmpeg available in PATH).",
    )
    preview_draw_poses: bool = Field(
        default=True,
        description="Overlay pose skeletons on preview videos when pose data is available.",
    )
    pose_conf_threshold: float = Field(
        default=0.2,
        description="Minimum pose keypoint confidence for visualization and analytics.",
    )
    use_patch_inference: bool = Field(
        default=False,
        description="Enable YOLO patch-based inference for high-recall detection.",
    )
    patch_shape_x: int = Field(
        default=960,
        description="Patch width when using patch-based inference.",
    )
    patch_shape_y: int = Field(
        default=960,
        description="Patch height when using patch-based inference.",
    )
    patch_overlap_x: int = Field(
        default=200,
        description="Horizontal overlap (pixels) between neighboring patches.",
    )
    patch_overlap_y: int = Field(
        default=200,
        description="Vertical overlap (pixels) between neighboring patches.",
    )
    patch_nms_threshold: float = Field(
        default=0.25,
        description="Patch-based NMS threshold when combining detections.",
    )
    reid_enabled: bool = Field(
        default=True, description="Enable ReID-assisted track stitching."
    )
    reid_similarity_threshold: float = Field(
        default=0.7, description="Cosine similarity threshold for merging tracks."
    )
    reid_time_gap_sec: float = Field(
        default=2.2,
        description="Maximum time gap (sec) between tracks considered for stitching.",
    )
    reid_model_name: str = Field(
        default="osnet_x1_0", description="Torchreid model name for feature extraction."
    )
    reid_model_path: str | None = Field(
        default=None, description="Optional custom checkpoint for Torchreid model."
    )
    video_output_dir: Path = Field(
        default=Path("runs/visualizations"),
        description="Directory where annotated preview videos will be stored.",
    )
    tracker_config_path: str = Field(
        default="bytetrack.yaml",
        description="Tracker configuration name/path for Ultralytics track API.",
    )
    detection_conf: float = Field(
        default=0.4, description="Confidence threshold for YOLO detections."
    )
    detection_iou: float = Field(
        default=0.5, description="IoU threshold for YOLO tracker associations."
    )
    detection_imgsz: int = Field(
        default=1280, description="Image size (short side) passed to YOLO track."
    )
    activity_model_path: str | None = Field(
        default="./yolo11n-pose.pt", description="Path to the action recognition weights."
    )
    activity_velocity_move_thresh: float = Field(
        default=140.0,
        description="Speed (px/sec) above which a person is confidently moving.",
    )
    activity_velocity_idle_thresh: float = Field(
        default=40.0,
        description="Speed (px/sec) below which a person is confidently idle.",
    )
    activity_min_interval_sec: float = Field(
        default=1.5,
        description="Shortest interval duration that will be recorded as activity.",
    )
    role_assignment_threshold: float = Field(
        default=0.6,
        description="Share of time in dominant zone required to fix person_type.",
    )
    zones: list[ZoneDefinition] = Field(
        default_factory=lambda: [
            ZoneDefinition(
                name="station_left",
                x_min=50,
                y_min=220,
                x_max=640,
                y_max=700,
                zone_category="station",
                person_type="operator",
            ),
            ZoneDefinition(
                name="station_right",
                x_min=650,
                y_min=220,
                x_max=1230,
                y_max=700,
                zone_category="station",
                person_type="operator",
            ),
            ZoneDefinition(
                name="corridor_main",
                x_min=0,
                y_min=80,
                x_max=1280,
                y_max=210,
                zone_category="corridor",
                person_type="supervisor",
            ),
            ZoneDefinition(
                name="visitor_strip",
                x_min=20,
                y_min=0,
                x_max=1260,
                y_max=70,
                zone_category="visitor",
                person_type="visitor",
            ),
            ZoneDefinition(
                name="restricted_zone",
                x_min=900,
                y_min=400,
                x_max=1200,
                y_max=650,
                zone_category="restricted",
                person_type="restricted",
            ),
        ],
        description="Rectangular production zones that drive roles and activities.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""

    # lru_cache гарантирует, что настройки считываются один раз за запуск процесса,
    # а дальнейшие импорты получают готовый объект без повторного чтения .env.
    return Settings()
