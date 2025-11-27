from pathlib import Path

import typer

from hackaton_system.db import init_db
from hackaton_system.db.seeder import seed_demo_data
from hackaton_system.pipeline import VideoProcessor

app = typer.Typer(help="Hackathon video analytics toolkit")


@app.command()
def setup_database() -> None:
    """Create database tables."""

    # Создаём таблицы и директории; вызывается один раз при развёртывании.
    init_db()
    typer.echo("Database initialized.")


@app.command()
def process_video(video_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Run the processing pipeline for a video file."""

    # Лениво создаём экземпляр пайплайна (внутри инициализируется БД и загрузчик YOLO).
    processor = VideoProcessor()
    video_id = processor.process_video(video_path)
    typer.echo(f"Video stored with id={video_id}")


@app.command()
def seed_demo(force: bool = typer.Option(False, "--force", help="Override existing data")) -> None:
    """Populate the database with deterministic demo rows."""

    # Загружаем предопределённые данные, чтобы интерфейс Streamlit сразу что-то показывал.
    rows = seed_demo_data(force=force)
    if rows:
        typer.echo(f"Inserted {rows} demo rows.")
    else:
        typer.echo("Demo data already present. Use --force to overwrite.")


if __name__ == "__main__":
    app()
