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

Схемы этапа 0 проверяют ID и metadata; полный валидатор геометрии/RLE,
индексирование, проверка уникальности и JSONL export — последующие этапы.

## Run manifest этапа 0

`manifest.json`: schema_version, seed, git_commit, git_status, Python/platform,
config_sha256 и явная отметка, что данные не проверялись (`dataset_hashes=null`).
`config.resolved.json`: snapshot с абсолютными локальными путями, не коммитится.
