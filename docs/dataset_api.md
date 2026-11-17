# Dataset API и визуальная проверка (этап 2)

API читает завершённый SQLite-кэш этапа 1. Исходные JSON повторно не сканируются;
незавершённый кэш не открывается. Ни PNG, ни аннотации, ни индекс не изменяются.

## Чтение

```python
from pathlib import Path
from actin_tracking.cli import resolve_config
from actin_tracking.data.dataset import FilamentDataset

config = resolve_config(Path("configs/datasets/v0.2.1.toml"), None)
with FilamentDataset.from_config(config, Path("outputs/stage1_audit"), "synthetic_val") as dataset:
    print(len(dataset))
    sequence_id = next(dataset.sequence_ids())
    for frame in dataset.get_sequence(sequence_id):
        print(frame.metadata.image_id, frame.image.shape)
        if frame.instances:
            instance = frame.instances[0]
            mask = instance.mask.decode()  # только здесь декодируется маска
            rle = instance.mask.to_rle()  # COCO compressed RLE, counts как str
        break
```

`get_frame(image_id)` принимает точный int или str. `__iter__` и `get_sequence`
возвращают генераторы; PNG открывается только при запросе соответствующего кадра.
Фабрика `from_config` выбирает один split и проверяет соответствие имени/source
индекса конфигурации. Для прямого открытия есть `FilamentDataset(index_path, images_root)`.
Контекстный менеджер закрывает read-only SQLite connection; открывать отдельный
экземпляр dataset для каждого процесса загрузки.

Каждый `FrameSample` содержит:

- `image`: grayscale `uint8[H,W]`; RGB/RGBA/палитровые 8-bit PNG явно переводятся в L.
  PNG читается один раз, его SHA256 проверяется против индекса. Изменённое изображение
  требует новой версии данных и аудита. 16-bit данные этим API не приводятся к uint8 молча.
- `instances`: tuple с точными строковыми instance/category/track IDs, bbox XYWH,
  исходным `original_bbox_xywh`, `iscrowd` и `LazyMask`.
- `metadata`: split, source, canonical image ID, физическое имя файла, все source
  image IDs, sequence ID, индекс кадра, window ID для inspection и nullable calibration.
  Числовые ID не проходят через float или JavaScript Number.
- `geometry`: исходный/current размер, шаги и XY-матрица преобразования.
- `valid_pixels`: bool-маска реальных пикселей; padding отмечен False.
- `photometric_history` и `dropped_instance_ids`: параметры преобразований
  и ID объектов, исчезнувших при lossy resize/crop.

FPS, µm/px и provenance остаются None для текущего набора. Скорость в физических
единицах не вычисляется.

## Real: кадры, окна и время

`len(real_eval)` и итерация дают **120 уникальных физических кадров**, а не 342
записи окон. `get_frame` по alias image ID возвращает каноническую запись того же
PNG; все aliases сохранены в `metadata.source_image_ids`. Конфликтующие GT и
невалидные PNG не допускаются. Индекс остаётся доступен для отдельного raw-аудита.

`get_sequence`/`sequence_ids` для real запрещены: непрерывность и GT trajectories
не подтверждены. `get_window(window_id)` даёт отсортированные кадры исходного окна
для просмотра, сохраняя `sequence_id=None`, `window_id` и
`temporal_continuity_confirmed=False`. Это не независимая видеовыборка.

## Геометрия и аугментации

```python
from actin_tracking.data.augmentations import Augmentations

augment = Augmentations(
    seed=20261117, resize=(512, 512), pad_to=(544, 560), crop_size=(512, 512),
    horizontal_flip_probability=0.5, brightness=(0.9, 1.1), noise_std=2,
)
with FilamentDataset.from_config(config, Path("outputs/stage1_audit"),
                                 "synthetic_val", augment) as dataset:
    frames = list(dataset.get_sequence(sequence_id))
```

Параметры выбираются stateless из SHA256(seed, split, sequence ID). Для одной
синтетической последовательности они одинаковы независимо от порядка чтения
или обращения через get_frame/get_sequence. Для одиночных real кадров ключом
является физическое имя. Для изменения аугментаций между эпохами меняйте seed
явно. Яркость/контраст не изменяют targets; Gaussian noise воспроизводим,
padding не изменяется фотометрическими операциями.

Явные операции доступны через `transform_frame(frame, tuple_of_GeometryOp)`:

- `GeometryOp("resize", (height, width))`: bilinear для изображения, nearest для масок;
- `GeometryOp("pad", (left, top, right, bottom))`;
- `GeometryOp("crop", (left, top, width, height))`: только внутри текущего кадра;
- `GeometryOp("hflip")`, `GeometryOp("vflip")`.

Преобразуются image, valid_pixels, bbox и маски. Исходный bbox не исправляется
по mask bbox: известное расхождение источника этапа 1 сохраняется явно.
`instance.mask.bbox_xywh()` позволяет отдельно получить tight mask bbox.
Resize/crop могут удалить тонкий или вышедший за ROI объект: для проверки
видимости декодируется одна маска за раз, массивы не кэшируются; ID удалённых
объектов записываются. Flip/pad сохраняют ленивость полностью.

Координаты XY относятся к границам пикселей (центр пикселя [0,0] — [0.5,0.5]).
Размеры массивов — H,W. `geometry.points(xy)`/`boxes(xywh)` переводят исходные
координаты в текущие; `inverse=True` переводит обратно. Clipping — явный параметр.

**Crop и downsampling теряют информацию.** Inverse-матрица обращает координаты,
но не восстанавливает отрезанные части bbox или исходные яркости.
`geometry.restore_mask(current_mask)` возвращает маску на исходной сетке и
boolean visibility. Отрезанная crop область отмечена False; нули там не следует
считать отрицательным GT. Оригинальная разметка остаётся в `original_bbox_xywh`
и `mask.original_rle`. Для полностью удалённых объектов полный GT остаётся
в SQLite и доступен при чтении исходного кадра без аугментаций.

## Collate

```python
from actin_tracking.data.samples import collate_frames, collate_sequences

batch = collate_frames(frames[:2])   # images [N,H,W]
temporal = collate_sequences([frames[:10], frames[10:]])  # images [B,T,H,W]
```

Батчи uint8, без PyTorch-зависимости и скрытой нормализации. Targets/metadata
находятся в `batch.frames` или `temporal.sequences`; маски остаются ленивыми.
Padding дописывается справа/снизу и входит в geometry history. Есть valid_pixels,
для временных батчей — valid_frames; отсутствующие временные slots не являются
пустыми размеченными кадрами.

Повторы физического кадра в батче по умолчанию запрещены; для намеренного
training sampling требуется `allow_duplicates=True`. Temporal collate требует
одну подтверждённую sequence на sample, последовательные frame indices и
одинаковую геометрию. Real окна, разрывы времени и разная аугментация кадров
отклоняются, а не превращаются в фиктивную непрерывную траекторию.

## Инспектор

```bash
uv run actin visualize --config configs/datasets/v0.2.1.toml \
  --index-dir outputs/stage1_audit --dataset real_eval --output outputs/real_preview

uv run actin visualize --config configs/datasets/v0.2.1.toml \
  --dataset synthetic_val --sequence-id 1221647990087 --limit 5 \
  --layers masks boxes skeleton tracks trajectories --output outputs/sequence_preview

uv run actin visualize --config configs/datasets/v0.2.1.toml \
  --dataset synthetic_val --sequence-id 1221647990087 --limit 2 \
  --resize 512 512 --pad-to 544 560 --crop-size 512 512 --hflip \
  --brightness 0.9 1.1 --noise-std 2 --output outputs/augmented_preview
```

Селекторы `--image-id`, `--sequence-id`, `--window-id` взаимоисключающие.
Без селектора выбираются уникальные физические кадры по SHA256(seed, filename).
`--limit` по умолчанию 3, seed 20261117. `--dry-run` проверяет параметры и
печатает конфигурацию без загрузки данных или записи файлов.

Слои: masks (alpha), boxes (source/current bbox), skeleton, tracks (цветные IDs),
trajectories. Цвет стабилен по sequence/identity; при отсутствии GT identities
показываются instance IDs. Текст ограничен `--max-labels` (20 по умолчанию), все
точные ID сохраняются в JSON. `--layers` без значений даёт изображение без разметки.

Скелет инспектора — `skimage.morphology.skeletonize(method="zhang")` в padded
foreground ROI, с возвратом в координаты кадра. Исторические диагностические
числа skeleton_pixels этапа 1 сохраняют прежний алгоритм и не переопределяются.
GT trajectories соединяют центры видимых масок только при одинаковом track ID
в соседних кадрах. Пропуски не интерполируются, неоднозначные IDs не соединяются; при смене
геометрии или обратном порядке кадров история сбрасывается.
Real trajectories запрещены до подтверждения GT и времени.

На кадр сохраняются `*_raw.png`, `*_overlay.png`, `*_comparison.png` и JSON:
raw здесь означает изображение в текущей геометрии до наложения GT. JSON содержит
исходный/current bbox, mask bbox, centroid, IDs, geometry и photometric history.
Есть config snapshot и manifest с dataset hash, code hashes, версиями библиотек,
статусом выполнения и mapping файлов. Существующий каталог не перезаписывается;
ошибочный частичный запуск отмечен failed, не complete.

[Проверенные локальные примеры](stage2_results.md).
