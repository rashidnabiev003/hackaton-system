## 🛠️ Кейс: детекция активности на промышленном видео

Документ описывает рабочий план хакатонного решения: от обработки видео до записи в базу и вывода аналитики в Streamlit.

---

## 1. Архитектура пайплайна

1. **Видеоинпут** → загрузка исходных роликов, метаданные (fps, длительность) кладём в таблицу `videos`.
2. **Детекция людей (YOLOv8/v11)** → выдаёт набор bounding boxes на каждый кадр.
3. **Трекинг (ByteTrack/BoT-SORT)** → объединяет детекции человека между кадрами, формирует `track_id`.
4. **Определение типа человека**:
   - либо дополнительным YOLO (детекция PPE/цветов) + правила,
   - либо light-классификатором (ResNet18) по кропам.
5. **Определение активности**:
   - fast-MVP: один YOLO, где класс = активность (`person_welding`, `person_idle`, …);
   - advanced: YOLO pose → LSTM/Transformer, либо video-classifier (SlowFast/X3D).
6. **Хранилище (PostgreSQL/SQLite)**: `videos`, `persons`, `detections`, `activities`.
7. **Streamlit-дашборд** → подключается к БД, строит графики: число людей во времени, время по активностям, heatmap “тип человека × активность”.

---

## 2. Технологический стек

| Блок | Инструменты |
| --- | --- |
| Детекция / трекинг | `ultralytics` (YOLOv8/YOLOv11), встроенный `track` CLI или `supervision` + ByteTrack |
| Обработка видео | OpenCV, ffmpeg-python |
| Классификация типа/активности | PyTorch (torchvision, pytorchvideo), scikit-learn/xgboost для классических моделей |
| База данных | PostgreSQL 15 (или SQLite на MVP), SQLAlchemy + Alembic для миграций |
| Дашборд | Streamlit + pandas + Plotly/Altair |

---

## 3. Структура хранилища

```sql
videos(video_id, filename, fps, duration_sec)
persons(person_id, video_id, track_id, person_type)
detections(id, video_id, person_id, frame_id, time_sec, x_min, y_min, x_max, y_max, confidence)
activities(id, video_id, person_id, activity_class, t_start_sec, t_end_sec, activity_conf)
pose_keypoints(id, video_id, person_id, frame_id, time_sec, keypoints, pose_conf)
```

*Логика:* при обработке видео создаём запись в `videos`, далее по каждому `track_id` вставляем `persons`, детекции и интервалы активности.

---

## 4. Пайплайн обработки (MVP)

1. `process_video(video)`:
   - прогоняем `yolo track ... tracker=bytetrack.yaml` → получаем JSON с рамками и `track_id`.
   - парсим результаты, заливаем `detections`.
2. `assign_person_type(track)`:
   - собираем кропы по треку → классифицируем роль (или ставим `unknown`).
3. `infer_activity(track)`:
   - вариант A (быстрый): класс активности приходит прямо из YOLO (детектор с несколькими активностями).
   - вариант B: по окнам по 32 кадра формируем клипы, гоняем через готовый 3D-классификатор.
4. `write_to_db(...)`:
   - объединяем соседние кадры с одним `activity_class` → диапазоны времени.

---

## 5. Streamlit-дашборд

Функциональность для защиты:

- селектор видео (данные из `videos`);
- график количества людей во времени;
- bar-chart суммарного времени по активностям;
- heatmap “тип работника × активность”;
- таблица с детальными эпизодами (`person_id`, `activity_class`, длительность).

Опционально: вывод кадров/клипов с наложенными рамками и подписями активности.

---

## 6. План работ на хакатон (2 дня)

**День 1**
1. Настройка окружения, скачивание весов YOLO, прогон демо-видео.
2. Подключение трекера, получение `track_id`.
3. Подготовка БД и скрипта записи в таблицы.

**День 2**
1. Быстрый классификатор активности (выбор подхода A/B).
2. Наполнение БД на одном-двух видео.
3. Создание Streamlit + графики.
4. Финальное демо: скрипт обработки + дашборд.

---

## 7. Идеи для развития (если останется время)

- Дообучить YOLO под спецусловия (ночь, каски, искры).
- Использовать YOLO pose → обучение action classifier по собственным меткам.
- Встроить модуль экспорта отчётов (PDF/Excel).
- В Streamlit добавить проигрывание видео с overlays и фильтры по диапазону времени.

---

## 8. Каркас реализации в репозитории

- CLI (см. `main.py`, Typer):
  - `python main.py setup-database` – создать таблицы.
  - `python main.py seed-demo` – заполнить демонстрационными данными (флаг `--force` перезаписывает).
  - `python main.py process-video path/to/video.mp4` – запускает обработку (детекции пока заглушены).
- Real-time / low-latency:
  - Используйте патчевый инференс (`HACKATON_USE_PATCH_INFERENCE=true`) + ByteTrack (параметры `HACKATON_TRACKER_TRACK_BUFFER`, `HACKATON_TRACKER_MATCH_THRESHOLD`).
  - На входе можно давать потоковое видеофайл/rtsp — ffmpeg-превью собирается через `HACKATON_PREVIEW_FFMPEG_PATH` (по умолчанию `ffmpeg` из PATH).
- Бэкенд:
  - `hackaton_system/config.py` – настройки (DB URL, пути к весам).
  - `hackaton_system/db/` – SQLAlchemy модели, сессии, сидер демо-данных.
  - `hackaton_system/pipeline/video_processor.py` – skeleton пайплайна (методы `_run_detection` / `_infer_activities` под ваши модели).
- Дашборд:
  - `streamlit run hackaton_system/dashboard/app.py`
  - `hackaton_system/dashboard/data_access.py` – загрузка данных из БД + плейсхолдеры.
  - Стилизация через кастомный CSS, графики Plotly, fallback на демо-данные, чтобы UI всегда показывал красивую панель.


## 9. Docker / Compose

Быстрый старт без локальной установки Python:

1. Собрать образ и запустить дашборд:

   ```bash
   docker compose up dashboard
   ```

   Приложение будет доступно на `http://127.0.0.1:8501`.

2. Все данные (SQLite, артефакты) сохраняются в `./data`, которая проброшена в контейнер.

3. Выполнить пайплайн/CLI-команды внутри контейнера можно через сервис `processor`, например:

   ```bash
   docker compose run --rm processor uv run python main.py seed-demo --force
   docker compose run --rm processor uv run python main.py process-video data/factory.mp4
   ```

Сборка использует `uv` для установки зависимостей, а системные библиотеки (`ffmpeg`, `libgl1`) уже включены, чтобы Ultralytics и OpenCV работали «из коробки».

## ?????????? ???????

- ?????????? ????? (COCO class 6) ???????????? ?? ?????? YOLO, ?? ??????????? ?? ?????? ????????.
- EasyOCR ???????? ?????? ????? ??????? ???????? ???????, ?????????? ?????????? ? 
uns/train_events/<video>_trains.json ? ???????????? ? preview.
- ?????????: HACKATON_ENABLE_TRAIN_DETECTION, HACKATON_TRAIN_NUMBER_LANGS, HACKATON_TRAIN_NUMBER_MIN_LENGTH.
