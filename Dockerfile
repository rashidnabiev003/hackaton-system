FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
LABEL authors="MIIREA NOC_AI LAB_Trust_AI"

WORKDIR /app

# System deps for OpenCV, Ultralytics, Streamlit rendering, ffmpeg video IO
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "hackaton_system/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

