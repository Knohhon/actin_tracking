# Проверка этапа 2: Dataset API и GT-инспектор

Дата: 17 ноября 2026. Ветка: `feat/stage2-dataset-api`.
Использован complete SQLite-кэш этапа 1; исходные изображения и аннотации не изменены.

## Проверки

- 39 тестов проекта, включая 16 новых проверок Dataset API/визуализации:
  точные IDs >2^53, lazy masks и итерация без preloading, числовой порядок,
  COCO RLE compressed/uncompressed/native roundtrip, согласованная геометрия,
  inverse points/boxes, restore masks с visibility после crop, удалённые targets,
  детерминированные sequence augmentations, batch padding, temporal validity,
  real dedup/window guards, изменённые PNG, CLI dry-run/экспорт и GT trajectories.
- Ruff, форматирование, mypy и установка по locked environment проходят.
- На настоящем real_eval прочитаны все 120 уникальных PNG с проверкой checksum;
  кадры не повторяются из-за окон. Metadata не объявляет их непрерывным видео.
- На настоящей synthetic val `1221647990087` загружены кадры 0..29 в правильном
  порядке; проверен temporal collate формы `[2,3,768,768]` с 5 действительными frames.

## Локальные артефакты

`outputs/stage2_preview/` содержит 16 кадров в пяти complete preview runs:

| Каталог | Кадры | Проверяемый режим |
| --- | ---: | --- |
| synthetic_train | 3 | Seeded случайные кадры: masks, boxes, skeleton, track IDs |
| synthetic_val | 3 | То же на val |
| real_eval | 3 | Уникальные физические PNG, instance IDs без фиктивных tracks |
| trajectories | 5 | Ordered synthetic frames 000..004, GT centroid trajectories |
| augmented | 2 | Resize 512² → pad 544×560 → crop 512² → hflip, brightness/noise |

Seed: 20261117. В каждом каталоге есть `manifest.json`, config snapshot,
raw/overlay/comparison PNG и per-frame JSON. Точные файлы, IDs, source hashes и
число instances перечислены в [компактном manifest](stage2_preview_manifest.json).
Большие PNG остаются локальными артефактами и не добавляются в Git.

Вручную просмотрены synthetic val `_029`, real `108.png`, augmented `_001` и
последний кадр GT trajectories `_004`: маски/скелеты совмещаются с изображением,
цвета identities сохраняются. Итоговые изображения совпадают с просмотренными
побитово; hashes сравнений сохранены в manifest. Известная неоднозначность source
bbox этапа 1 остаётся видимой и отражена отдельными source/current/mask boxes.

В augmented примерах из 285/284 source instances осталось 282/279: 3/5 объектов
исчезли после lossy геометрии. Их точные IDs записаны в dropped_instance_ids;
исходный GT остаётся в SQLite. Это изменение текущих targets, а не исправление данных.

[Сравнение synthetic val](../outputs/stage2_preview/synthetic_val/0000_925c120f3e66_comparison.png),
[real](../outputs/stage2_preview/real_eval/0000_2378926a9bcc_comparison.png),
[аугментации](../outputs/stage2_preview/augmented/0001_438d14d85e70_comparison.png),
[траектории](../outputs/stage2_preview/trajectories/0004_40179c1180e8_comparison.png).
Ссылки предназначены для локального checkout с выполненными previews.

## Границы реализации

FPS и µm/px неизвестны; физические скорости не вычисляются. Real GT trajectories
недоступны. Оригинальные bbox сохраняются без автоматической нормализации.
Crop/downsampling необратимо теряют информацию; visibility отмечает область,
неизвестную после crop. Диагностические proxy длины этапа 1 не переопределяются
скелетизацией инспектора. Нормализацию tensor/channel layout и работу GPU
выполнят будущие model adapters; Dataset API возвращает grayscale uint8.
