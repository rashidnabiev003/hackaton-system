from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict, cast

import pandas as pd
import plotly.express as _px  # type: ignore[import]
import streamlit as _st  # type: ignore[import]

from hackaton_system.config import get_settings
from hackaton_system.dashboard.data_access import (
    delete_video_and_related,
    list_available_videos,
    load_activity_summary,
    load_headcount,
    load_person_episodes,
    load_role_activity_matrix,
)
from hackaton_system.pipeline import VideoProcessor

px = cast(Any, _px)
st = cast(Any, _st)
settings = get_settings()
PREVIEW_DIR = getattr(settings, "video_output_dir", Path("runs/visualizations"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)


def _process_video_file(target_path: Path) -> None:
    status_box = st.sidebar.empty()
    try:
        with status_box, st.spinner("Запускаем пайплайн..."):
            processor = VideoProcessor()
            video_id = processor.process_video(target_path)
        preview_candidate = PREVIEW_DIR / f"{video_id}_{target_path.stem}.mp4"
        if preview_candidate.exists():
            status_box.success(
                f"Готово! video_id={video_id}. Визуализация: {preview_candidate.name}"
            )
        else:
            status_box.warning(
                f"Готово! video_id={video_id}, но превью не найдено. Проверьте журналы консоли."
            )
        st.rerun()
    except Exception as exc:  # pragma: no cover - интерактивная ошибка
        status_box.error(f"Ошибка обработки: {exc}")


class VideoRecord(TypedDict):
    id: int | None
    filename: str
    duration_sec: float
    fps: float


st.set_page_config(
    page_title="Industrial Activity Analytics",
    page_icon="🛠️",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, #152238 0%, #0b1220 45%, #04060b 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 1rem;
        background: linear-gradient(120deg, rgba(14,165,233,0.25), rgba(16,185,129,0.25));
        border: 1px solid rgba(226,232,240,0.1);
        backdrop-filter: blur(6px);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        padding: 1rem 1.25rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(226,232,240,0.12);
        background: rgba(15,23,42,0.65);
        box-shadow: 0px 10px 40px rgba(15,23,42,0.35);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .section-title {
        font-size: 1.1rem;
        letter-spacing: 0.05rem;
        color: #cbd5f5;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar — сначала даём возможность запустить обработку, потом выбор готового ролика
RAW_VIDEO_DIR = Path("video")
RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.sidebar.header("Обработка видео")
uploaded_video = st.sidebar.file_uploader(
    "Загрузите файл", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False
)
process_clicked = st.sidebar.button("Запустить обработку", width="stretch")
if process_clicked:
    if uploaded_video is None:
        st.sidebar.warning("Сначала загрузите файл, затем запускайте обработку.")
    else:
        target_path = RAW_VIDEO_DIR / uploaded_video.name
        with open(target_path, "wb") as dst:
            dst.write(uploaded_video.getbuffer())
        _process_video_file(target_path)

st.sidebar.header("Файлы на сервере")
raw_files = sorted(
    [p for p in RAW_VIDEO_DIR.glob("*") if p.is_file()],
    key=lambda p: p.name.lower(),
)
if raw_files:
    raw_names = [file.name for file in raw_files]
    selected_raw = st.sidebar.selectbox("Загруженные файлы", raw_names)
    selected_raw_path = RAW_VIDEO_DIR / selected_raw
    size_mb = selected_raw_path.stat().st_size / (1024 * 1024)
    st.sidebar.caption(f"Размер: {size_mb:.2f} MB")
    if st.sidebar.button("Обработать выбранный файл", key="process_existing_raw"):
        _process_video_file(selected_raw_path)
    if st.sidebar.button("Удалить файл", key="delete_raw_file"):
        try:
            selected_raw_path.unlink()
            st.sidebar.success("Файл удалён.")
            st.rerun()
        except Exception as exc:  # pragma: no cover - UI feedback
            st.sidebar.error(f"Не удалось удалить файл: {exc}")
else:
    st.sidebar.info("Нет загруженных роликов. Добавьте файл через форму выше.")

st.sidebar.header("Просмотр результатов")
videos_df: pd.DataFrame = list_available_videos()
video_records: List[VideoRecord] = cast(
    List[VideoRecord],
    cast(Any, videos_df).to_dict(orient="records"),
)
video_options: Dict[str, int | None] = {
    record["filename"]: record.get("id") for record in video_records
}
selected_filename: str = st.sidebar.selectbox(
    "Выберите ролик", list(video_options.keys())
)
selected_video_id: int | None = video_options[selected_filename]

selected_meta = next(
    (record for record in video_records if record["filename"] == selected_filename),
    None,
)
if selected_meta:
    minutes = selected_meta["duration_sec"] / 60
    st.sidebar.metric("Длительность", f"{minutes:.1f} мин")
    st.sidebar.metric("FPS", f"{selected_meta['fps']:.1f}")
    if selected_video_id is not None and st.sidebar.button(
        "Удалить обработанные данные", type="secondary"
    ):
        removed = delete_video_and_related(selected_video_id)
        if removed:
            st.sidebar.success("Видео и связанные данные удалены.")
        else:
            st.sidebar.warning("Запись не найдена или уже удалена.")
        st.rerun()
else:
    st.sidebar.info("Пока нет обработанных видео — отображаются демо-данные.")


def _render_metric(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>🏭 Панель мониторинга активности</h1>
        <p>Аналитика по людям, ролям и действиям на производственной площадке.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

headcount_df = load_headcount(selected_video_id)
activity_summary = load_activity_summary(selected_video_id)
role_matrix = load_role_activity_matrix(selected_video_id)
episodes_df = load_person_episodes(selected_video_id)

working_minutes = (
    float(
        activity_summary.loc[
            activity_summary["activity_class"] == "working", "duration_min"
        ].sum()
    )
    if not activity_summary.empty
    else 0.0
)
idle_minutes = (
    float(
        activity_summary.loc[
            activity_summary["activity_class"] == "idle_at_station", "duration_min"
        ].sum()
    )
    if not activity_summary.empty
    else 0.0
)
restricted_minutes = (
    float(
        activity_summary.loc[
            activity_summary["activity_class"] == "in_restricted_zone", "duration_min"
        ].sum()
    )
    if not activity_summary.empty
    else 0.0
)
unique_people = episodes_df["person_id"].nunique() if not episodes_df.empty else 0

col1, col2, col3 = st.columns(3)
with col1:
    _render_metric("Работа", f"{working_minutes:.1f} мин")
with col2:
    _render_metric("Простой", f"{idle_minutes:.1f} мин")
with col3:
    summary_caption = f"{unique_people} чел"
    if restricted_minutes > 0:
        summary_caption += f" · {restricted_minutes:.1f} мин в запрете"
    _render_metric("Всего людей / нарушения", summary_caption)

preview_path: Path | None = None
if selected_video_id:
    candidate = PREVIEW_DIR / f"{selected_video_id}_{Path(selected_filename).stem}.mp4"
    if candidate.exists():
        preview_path = candidate

st.markdown('<div class="section-title">Видео с разметкой</div>', unsafe_allow_html=True)
if preview_path and preview_path.exists():
    with open(preview_path, "rb") as preview_file:
        st.video(preview_file.read(), format="video/mp4")
else:
    st.info("Пока нет визуализации для этого ролика.")

# Charts section
st.markdown(
    '<div class="section-title">Динамика людей на объекте</div>', unsafe_allow_html=True
)
if not headcount_df.empty:
    headcount_fig = px.area(
        headcount_df,
        x="time_sec",
        y="headcount",
        color_discrete_sequence=["#22d3ee"],
        labels={"time_sec": "Секунды", "headcount": "Люди"},
    )
    headcount_fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(headcount_fig, width="stretch")
else:
    st.info("Нет данных по трекам — запустите пайплайн обработки.")

left, right = st.columns(2)

with left:
    st.markdown(
        '<div class="section-title">Время по активностям</div>', unsafe_allow_html=True
    )
    if not activity_summary.empty:
        duration_fig = px.bar(
            activity_summary,
            x="activity_class",
            y="duration_min",
            color="activity_class",
            text_auto=".1f",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"activity_class": "Активность", "duration_min": "Минуты"},
        )
        duration_fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(duration_fig, width="stretch")
    else:
        st.info("Добавьте активности, чтобы увидеть распределение времени.")

with right:
    st.markdown(
        '<div class="section-title">Матрица “роль × активность”</div>',
        unsafe_allow_html=True,
    )
    if not role_matrix.empty:
        heatmap_source = role_matrix.set_index("person_type")
        heatmap_fig = px.imshow(
            heatmap_source,
            color_continuous_scale="viridis",
            labels=dict(x="Активность", y="Тип сотрудника", color="Минуты"),
            aspect="auto",
        )
        heatmap_fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(heatmap_fig, width="stretch")
    else:
        st.info("Пока нет данных для построения матрицы.")

st.markdown('<div class="section-title">Хронология движения по трекам</div>', unsafe_allow_html=True)
motion_df = episodes_df[episodes_df["activity_class"] == "moving"].copy()
if motion_df.empty and not episodes_df.empty:
    motion_df = episodes_df.copy()
if not motion_df.empty:
    motion_df = motion_df.rename(columns={"t_start_sec": "start_sec", "t_end_sec": "end_sec"})
    motion_df["duration_sec"] = motion_df["end_sec"] - motion_df["start_sec"]
    motion_df["person_label"] = motion_df.apply(
        lambda row: f"{row['person_type']} · ID {row['track_id']}"
        if isinstance(row.get("person_type"), str)
        else f"Track {row['track_id']}",
        axis=1,
    )
    timeline_fig = px.bar(
        motion_df,
        x="duration_sec",
        y="person_label",
        base="start_sec",
        color="activity_class",
        orientation="h",
        labels={
            "duration_sec": "Длительность, сек",
            "start_sec": "Начало, сек",
            "person_label": "Сотрудник / Track",
            "activity_class": "Активность",
        },
    )
    timeline_fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Секунды от начала видео",
    )
    st.plotly_chart(timeline_fig, width="stretch")
else:
    st.info("Нет данных о движении — обработайте ролик, чтобы увидеть таймлайн.")

st.markdown('<div class="section-title">Таблица эпизодов</div>', unsafe_allow_html=True)
if not episodes_df.empty:
    role_options = sorted(episodes_df["person_type"].dropna().unique().tolist()) or [
        "unknown"
    ]
    activity_options = sorted(
        episodes_df["activity_class"].dropna().unique().tolist()
    ) or ["walking"]
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_roles = st.multiselect(
            "Тип сотрудника",
            role_options,
            default=role_options,
        )
    with filter_col2:
        selected_activities = st.multiselect(
            "Активность",
            activity_options,
            default=activity_options,
        )
    filtered = episodes_df[
        episodes_df["person_type"].isin(selected_roles)
        & episodes_df["activity_class"].isin(selected_activities)
    ].copy()
    filtered["duration_sec"] = filtered["duration_sec"].round(1)
    filtered = filtered.rename(
        columns={
            "person_id": "Person ID",
            "track_id": "Track ID",
            "person_type": "Тип",
            "activity_class": "Активность",
            "t_start_sec": "Начало, сек",
            "t_end_sec": "Конец, сек",
            "duration_sec": "Длительность, сек",
        }
    )
    st.dataframe(
        filtered[
            [
                "Person ID",
                "Track ID",
                "Тип",
                "Активность",
                "Начало, сек",
                "Конец, сек",
                "Длительность, сек",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info(
        "Нет записанных эпизодов — выполните обработку видео или загрузите демо-данные."
    )
