FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

LABEL authors="MIIREA NOC_AI LAB_Trust_AI"

WORKDIR /app

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code
COPY . .

EXPOSE 8501

CMD ["uv", "run", "dashboard"]
