from __future__ import annotations

from typing import Any, Dict, List, TypedDict, cast

import pandas as pd
import plotly.express as _px  # type: ignore[import]
import streamlit as _st  # type: ignore[import]

from hackaton_system.dashboard.data_access import (
    list_available_videos,
    load_activity_summary,
    load_headcount,
    load_person_episodes,
    load_role_activity_matrix,
)

px = cast(Any, _px)
st = cast(Any, _st)


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

# Sidebar — приводим датафрейм к TypedDict, чтобы типы были явные
st.sidebar.header("Видео")
videos_df: pd.DataFrame = list_available_videos()
video_records: List[VideoRecord] = cast(
    List[VideoRecord],
    cast(Any, videos_df).to_dict(orient="records"),
)
video_options: Dict[str, int | None] = {
    record["filename"]: record.get("id") for record in video_records
}
selected_filename: str = st.sidebar.selectbox("Выберите ролик", list(video_options.keys()))
selected_video_id: int | None = video_options[selected_filename]

selected_meta = next((record for record in video_records if record["filename"] == selected_filename), None)
if selected_meta:
    minutes = selected_meta["duration_sec"] / 60
    st.sidebar.metric("Длительность", f"{minutes:.1f} мин")
    st.sidebar.metric("FPS", f"{selected_meta['fps']:.1f}")
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

working_minutes = float(
    activity_summary.loc[
        activity_summary["activity_class"] == "working", "duration_min"
    ].sum()
) if not activity_summary.empty else 0.0
idle_minutes = float(
    activity_summary.loc[
        activity_summary["activity_class"] == "idle_at_station", "duration_min"
    ].sum()
) if not activity_summary.empty else 0.0
restricted_minutes = float(
    activity_summary.loc[
        activity_summary["activity_class"] == "in_restricted_zone", "duration_min"
    ].sum()
) if not activity_summary.empty else 0.0
unique_people = episodes_df["person_id"].nunique() if not episodes_df.empty else 0

col1, col2, col3 = st.columns(3)
with col1:
    _render_metric("Работа у станков", f"{working_minutes:.1f} мин")
with col2:
    _render_metric("Простой у станков", f"{idle_minutes:.1f} мин")
with col3:
    summary_caption = f"{unique_people} чел"
    if restricted_minutes > 0:
        summary_caption += f" · {restricted_minutes:.1f} мин в запрете"
    _render_metric("Всего людей / нарушения", summary_caption)

# Charts section
st.markdown('<div class="section-title">Динамика людей на объекте</div>', unsafe_allow_html=True)
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
    st.plotly_chart(headcount_fig, use_container_width=True)
else:
    st.info("Нет данных по трекам — запустите пайплайн обработки.")

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">Время по активностям</div>', unsafe_allow_html=True)
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
        st.plotly_chart(duration_fig, use_container_width=True)
    else:
        st.info("Добавьте активности, чтобы увидеть распределение времени.")

with right:
    st.markdown('<div class="section-title">Матрица “роль × активность”</div>', unsafe_allow_html=True)
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
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("Пока нет данных для построения матрицы.")

st.markdown('<div class="section-title">Таблица эпизодов</div>', unsafe_allow_html=True)
if not episodes_df.empty:
    role_options = sorted(episodes_df["person_type"].dropna().unique().tolist()) or ["unknown"]
    activity_options = sorted(episodes_df["activity_class"].dropna().unique().tolist()) or [
        "walking"
    ]
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
            "person_type": "Тип",
            "activity_class": "Активность",
            "t_start_sec": "Начало, сек",
            "t_end_sec": "Конец, сек",
            "duration_sec": "Длительность, сек",
        }
    )
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Нет записанных эпизодов — выполните обработку видео или загрузите демо-данные.")
