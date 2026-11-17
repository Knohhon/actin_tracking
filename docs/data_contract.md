# Контракт данных v1

Канонические records определены в `src/actin_tracking/data/schemas.py`.

- Идентификаторы сохраняются строками; исходный Python int преобразуется точно.
  Float/bool IDs запрещены, включая значения меньше 2^53. Нельзя выполнять
  промежуточное преобразование ID в float/JavaScript Number.
- `instance_id` уникален в кадре, `track_id` — identity внутри sequence;
  отсутствие identity обозначается null, не нулём.
- `bbox_xywh`, `center_xy` и маски относятся к оригинальному кадру, начало
  координат слева сверху. Размер RLE `[height, width]`, COCO column-major.
- Предсказания: JSONL, одна запись FilamentInstance на объект, RLE counts —
  JSON-строка (не bytes). Версия схемы указывается в manifest запуска.
- `SequenceMetadata`: source synthetic/real, nullable fps/pixel_size_um;
  известная калибровка положительна, конечна и имеет provenance.
- Неизвестные score, length, center и calibration — null, а не 0.
- Synthetic: группировать по sequence_id и сортировать по числовому суффиксу.
  Real windows — не независимые съемки; оценка по уникальным PNG.
- Исходные данные неизменяемы. Преобразования создают новые версии с provenance.

Схемы этапа 0 проверяют ID и metadata; аудит этапа 1 проверяет RLE/геометрию и уникальность и создаёт SQLite-индекс.
JSONL export предсказаний появится с evaluator.

## Run manifest этапа 0

`manifest.json`: schema_version, seed, git_commit, git_status, Python/platform,
config_sha256 и явная отметка, что данные не проверялись (`dataset_hashes=null`).
`config.resolved.json`: snapshot с абсолютными локальными путями, не коммитится.

## SQLite audit index

`images`: точный TEXT id, file_name, sequence_id, числовой frame_index, размеры
и исходный JSON payload. `annotations`: row_id, точные TEXT id/image_id/track_id,
category_id, payload, хэши масок и ключ детерминированной выборки. Индексы на ID,
image_id и (sequence_id,frame_index). `physical_frames`: file_name → первая
каноническая запись, conflicted/valid flags, SHA256 и размер PNG. Конфликтующие
версии GT сохраняются; canonical не означает автоматическое исправление.
`issues` хранит все замечания, `metadata` — summary завершённого импорта.

Повреждённый JSON прерывает импорт; частичный индекс не содержит summary.
Не используйте незавершённые индексы для evaluation. Real alternative источник
сравнивается с основным по per-frame multiset GT без source-local IDs и никогда
не добавляет новые независимые evaluation кадры.


## Frame/Dataset contract этапа 2

`FilamentDataset` читает complete SQLite, grayscale PNG и lazy RLE targets.
FrameMetadata содержит точные строковые image/source-image/sequence/window IDs;
для real sequence_id=None, temporal_continuity_confirmed=False. Unknown calibration
остаётся None. Source bbox сохраняется отдельно от transformed/clipped bbox;
mask-derived bbox получается явно, исходники не нормализуются молча.

GeometryTrace использует XY edge coordinates и размеры H,W, поддерживает
resize/pad/crop/flip и inverse coordinates. Restore raster после crop возвращает
visibility: отброшенные пиксели неизвестны. Пустые после resize/crop instances
удаляются из текущих targets с записью ID в dropped_instance_ids.
Batch padding имеет valid_pixels/valid_frames и полную историю координат.
Полный интерфейс и примеры: [dataset_api.md](dataset_api.md).
