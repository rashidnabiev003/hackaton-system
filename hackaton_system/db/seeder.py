from __future__ import annotations

import logging
from random import Random

from hackaton_system.db import Activity, Detection, Person, Video, get_session, init_db

LOGGER = logging.getLogger(__name__)


def seed_demo_data(force: bool = False) -> int:
    """Populate the database with deterministic demo rows for dashboards."""

    init_db()
    rng = Random(42)

    with get_session() as session:
        video_count = session.query(Video).count()
        if video_count and not force:
            LOGGER.info("Database already contains data. Skipping demo seeding.")
            return 0

        if force:
            session.query(Activity).delete()
            session.query(Detection).delete()
            session.query(Person).delete()
            session.query(Video).delete()
            session.flush()

        # создаём ролик-демо, чтобы дашборд всегда мог показать статистику
        video = Video(filename="demo_factory.mp4", fps=25.0, duration_sec=300)
        session.add(video)
        session.flush()

        person_types = ["operator", "supervisor", "visitor"]
        activity_blocks = [
            ["working", "idle_at_station", "walking"],
            ["walking", "standing", "walking"],
            ["walking", "in_restricted_zone", "standing"],
        ]
        base_boxes = {
            "operator": (180, 260, 320, 520),
            "supervisor": (100, 120, 220, 260),
            "visitor": (940, 420, 1080, 620),
        }

        total_rows = 0
        for track_id, person_type in enumerate(person_types, start=1):
            # фиксируем человека + его "профессию"
            person = Person(
                video_id=video.id,
                track_id=track_id,
                person_type=person_type,
                person_type_conf=0.9,
            )
            session.add(person)
            session.flush()

            frame = 0
            x_min, y_min, x_max, y_max = base_boxes[person_type]
            while frame <= 300:
                # имитируем, что трекер отдаёт рамки каждые 15 кадров
                time_sec = frame / video.fps
                bbox = (
                    x_min + rng.randint(-5, 5),
                    y_min + rng.randint(-5, 5),
                    x_max + rng.randint(0, 5),
                    y_max + rng.randint(0, 5),
                )
                detection = Detection(
                    video_id=video.id,
                    person_id=person.id,
                    frame_id=frame,
                    time_sec=time_sec,
                    x_min=bbox[0],
                    y_min=bbox[1],
                    x_max=bbox[2],
                    y_max=bbox[3],
                    confidence=0.8,
                )
                session.add(detection)
                frame += 15
                total_rows += 1

            blocks = activity_blocks[track_id - 1]
            block_duration = video.duration_sec / len(blocks)
            for idx, activity in enumerate(blocks):
                # делим всю длительность видео на блоки активности
                start = idx * block_duration
                end = start + block_duration - 1
                session.add(
                    Activity(
                        video_id=video.id,
                        person_id=person.id,
                        activity_class=activity,
                        t_start_sec=start,
                        t_end_sec=end,
                        activity_conf=0.75,
                    )
                )
                total_rows += 1

        LOGGER.info("Seeded demo data: %s rows", total_rows)
        return total_rows
