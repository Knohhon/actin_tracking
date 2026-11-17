# Actin Tracking — план реализации и целевое состояние проекта

**Статус:** проектный план, не отчёт о выполненной реализации  
**Дата:** 17 ноября 2026  
**Основной репозиторий:** https://github.com/Knohhon/actin_tracking  
**Описание исходных данных:** https://github.com/Knohhon/actin_tracking/blob/main/docs/data_overview.md  
**Оригинальный ATLAS:** https://github.com/obilogy/ATLAS  
**Статья:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12271436/  
**Данные и релиз ATLAS:** https://zenodo.org/records/15320358

## 1. Цель проекта

Создать воспроизводимую систему автоматической разметки последовательностей изображений актиновых филаментов, которая:

1. Детектирует отдельные филаменты на кадре.
2. Определяет их контуры (instance segmentation).
3. Сопоставляет филаменты между кадрами и формирует траектории (`track_id`).
4. Измеряет длину, положение и, **если известна временная и пространственная калибровка**, скорость движения.
5. Позволяет честно сравнить classical CV, обучаемые модели и оригинальный/воспроизведённый ATLAS на одних данных и метриках.
6. Сохраняет предсказания, метрики, параметры, визуализации и информацию о версиях кода/моделей для повторения эксперимента.

**Разделение целей:** (a) воспроизводимое сравнение бейзлайнов, (b) научно корректная репликация ATLAS, (c) выбор лучшего практического метода для автoразметки. Репликация ATLAS не обязана оказаться лучшим методом на конкретных данных.

## 2. Известное состояние данных и ограничения

Источник: `docs/data_overview.md` (осмотр набора `data/v0.2.1/` от 16.11.2026). Эти цифры описывают исходный набор; корректность всех аннотаций **ещё не проверена**.

| Поднабор | Кадры | Предполагаемая организация | Размер | Разметка |
|---|---:|---|---|---|
| Synthetic train | 20 100 | 670 последовательностей × 30 | 768×768 | COCO RLE instance masks, boxes, `track_id` в просмотренных записях |
| Synthetic val | 300 | 10 последовательностей × 30 | 768×768 | То же, наличие/целостность всех ID требует аудита |
| Real | 120 уникальных PNG | Три диапазона `100–139`, `200–239`, `300–339`; временная непрерывность не подтверждена | 512×512 | COCO RLE instance masks и boxes; в просмотренных записях `track_id` нет |

**Специфика форматов:**

- Обучающий JSON ≈ 3.179 GB: **не загружать целиком через обычный `json.load` при аудите**; использовать потоковый парсер (`ijson`) и/или однократную конвертацию в индексированный формат.
- Маски хранятся в COCO RLE (`size`, `counts`); для декодирования использовать `pycocotools`.
- Синтетические `images` перемешаны: группировать по `sequence_id`, сортировать по индексу в имени файла, не по порядку записей JSON.
- Числовые ID могут быть больше `2^53-1`: обрабатывать как точные Python `int` или строки; не выполнять round-trip через JavaScript `Number`, не преобразовывать без проверки в float.
- В `seq_v1_val_real.rle.json` указано 342 вхождения кадров в 114 перекрывающихся окнах длины 3, но **120 уникальных PNG**; использовать уникальные физические кадры в оценке frame-level метрик.
- В `val_real_seqlen3.rle.json` — 117 уникальных кадров без явного `sequence_id`; это **не отдельная независимая выборка**. Три изображения `139.png`, `239.png`, `339.png` в ней отсутствуют.
- Для реальных данных возможна проверка detection/segmentation; tracking-метрики по GT недоступны, пока не подтверждены или не созданы межкадровые идентификаторы.
- FPS, размер пикселя в µm, согласованность реальных диапазонов и условия лицензии необходимо выяснить отдельно. При отсутствии калибровки результаты только в `px`, `px/frame`.
- Исходные изображения и аннотации считать **неизменяемыми**. Все исправления и преобразования — отдельными версиями с provenance.

## 3. Зафиксированное целевое состояние (Definition of Done всего проекта)

После выполнения всех этапов репозиторий должен содержать:

- **Data layer:** единый проверенный, документированный API для synthetic и real, потоковый импорт исходных аннотаций, индексы для выборочного чтения, валидатор, dataset manifest, непересекающиеся splits и точные идентификаторы.
- **Пять воспроизводимых семейств моделей:** `B0 classical threshold`, `B1 ridge`, `B2 U-Net + postprocessing`, `B3 instance-segmentation detector`, `B4 ATLAS`. Для ATLAS — отдельно `official` (обёртка над оригиналом) и `reimplementation` (своя модульная реализация); не смешивать эти результаты.
- **Общие интерфейсы:** `Detector`, `Segmenter`, `Tracker`, `Measurement`, `Pipeline`, `Evaluator`; независимые слои обработки кадров, масок, связей, физических измерений.
- **Единый evaluator:** обнаружение, instance segmentation, tracking (только где есть GT identities), измерения длины/скорости (только где есть валидный GT и калибровка), скорость выполнения и память; строгий протокол matching.
- **Набор oracle/ablation экспериментов:** `GT masks → simple tracker`, `GT boxes → DeepSORT`, `predicted masks → GT associations` (последний — только в диагностическом evaluation, не как deployable method), компоненты ATLAS по отдельности.
- **Конфигурируемый CLI:** audit, preprocess/index, train, infer, evaluate, benchmark, visualize, report; одинаковый запуск с локальными путями к данным.
- **Артефакты:** предсказания в нормализованном machine-readable формате, checkpoints/их ссылки и хэши, experiment configs, seed, git commit, метрики CSV/JSON, ошибки и примеры, итоговый сравнительный отчёт с таблицами и графиками.
- **Тесты и документация:** unit/integration/smoke tests, README со сценарием запуска, data schema, runbook для ATLAS, предупреждения о licence/data leakage и dataset limitations.
- **Практический результат:** можно передать новую последовательность и получить per-frame masks, boxes, tracks и визуализированное видео, а также длины/скорости, если для них достаточно данных.

**Критерий приемки:** другой разработчик в чистом окружении может по README выполнить smoke run, затем воспроизвести сравнение на фиксированной выборке и получить согласованные метрики в пределах заранее указанного допуска (с сохранёнными версиями зависимостей).

### 3.1. Внешний контракт результатов

Предлагаемый канонический формат результата, независимо от модели:

```python
@dataclass
class FilamentInstance:
    sequence_id: str
    frame_index: int
    instance_id: str       # уникален в пределах кадра
    bbox_xywh: tuple[float, float, float, float]
    mask_rle: dict | None  # COCO RLE, сохранение без раздувания памяти
    score: float | None
    track_id: str | None   # постоянный в пределах последовательности
    center_xy: tuple[float, float] | None
    length_px: float | None

@dataclass
class SequenceMetadata:
    sequence_id: str
    source: str            # synthetic / real
    fps: float | None
    pixel_size_um: float | None
    calibration_provenance: str | None
```

Внутри вычислений допустимы массивы NumPy; для сохранения использовать COCO RLE/Parquet и таблицы. Вся геометрия выполняется в координатах **оригинального кадра**, даже если модель использует resize/pad/crop. `track_id` не заменяет `instance_id`. `None` означает действительно неизвестное поле, а не ноль.

### 3.2. Предлагаемая целевая структура репозитория

```text
actin_tracking/
├── README.md
├── pyproject.toml
├── uv.lock                 # либо другой зафиксированный lockfile
├── configs/
│   ├── datasets/
│   ├── models/{b0,b1,b2,b3,atlas_official,atlas_reimpl}/
│   └── experiments/
├── data/                   # исходные большие данные вне Git; manifests в Git
├── src/actin_tracking/
│   ├── cli.py
│   ├── data/{schemas,readers,indexing,validation,splits}.py
│   ├── preprocessing/
│   ├── detection/
│   ├── segmentation/
│   ├── tracking/
│   ├── measurements/
│   ├── pipelines/
│   ├── evaluation/
│   ├── visualization/
│   └── reporting/
├── scripts/
├── experiments/            # протоколы и configs; не коммитить большие результаты
├── tests/{unit,integration,smoke}/
├── docs/{data_overview,data_contract,benchmark_protocol,atlas_reproduction}.md
└── outputs/                # игнорируется Git; результаты по run_id
```

Структура — **предлагаемая**, не утверждение о существующем коде. Не плодить файлы до появления реальной потребности.

## 4. Протокол оценки (зафиксировать ДО обучения)

### 4.1. Датасеты и split policy

1. `synthetic_train`: обучение и настройка классических методов на training-subset, если они требуют порогов.
2. `synthetic_val`: основная фиксированная синтетическая валидация, без пересечения `sequence_id` с train; не использовать как скрытый train.
3. `real_eval`: уникальные 120 кадров, без повторов окон; если real используется для fine-tuning, **сначала** создать заранее зафиксированные раздельные группы для обучения/валидации/теста на уровне независимых исходных съемок, если независимость подтверждена. В противном случае любые результаты после адаптации отмечать как exploratory, не как unbiased test.
4. Если исходная разметка реальных изображений содержит разные версии одного кадра, выбрать каноническую запись, проверить согласованность дубликатов, задокументировать конфликтные instances.
5. Не объединять три диапазона реальных файлов в одну непрерывную временную траекторию без подтверждения.
6. В синтетике проверить, что одинаковые исходные траектории/изображения не попали в разные splits через генерацию.

### 4.2. Метрики

| Уровень | Метрики | Условия |
|---|---|---|
| Detection | Precision/Recall/F1 при фиксированном IoU, AP50 и mAP50:95 | фиксированные thresholds и matching |
| Instance segmentation | mask AP, mask IoU/Dice после matching, FN/FP, качество на пересечениях | semantic IoU отдельно от instance metrics |
| Tracking | HOTA, IDF1, ID switches, fragmentation, Mostly Tracked | только GT `track_id`, и непрерывные последовательности |
| Длина | MAE, median AE, bias, error by filament length | GT mask/skeleton и метод подсчёта длины должны совпадать |
| Скорость | MAE, bias, error distribution | GT trajectories, FPS/µm-пиксель или согласованный px/frame |
| Runtime | ms/frame, peak RAM/VRAM, throughput | одинаковая машина, фиксированные условия |

Важно: раздельно агрегировать **micro** (по объектам/кадрам) и **macro** (по последовательностям), показывать `N` и 95% CI bootstrap по независимым последовательностям там, где данных хватает. На реальных данных 3 возможные группы — недостаточная основа для уверенных CI по независимым видео; не имитировать точность. Отдельные срезы: плотность, размер/длина, контраст/SNR, пересечения, граница кадра, тип данных.

Правило честного сравнения: одинаковые разделения, обработка координат, правила исключения edge objects, конфигурация confidence и одинаковый evaluator. Публиковать **raw/unfiltered** метрики и ATLAS-compatible filtered metrics отдельно, чтобы удаление сложных объектов не выглядело как улучшение без снижения coverage.

## 5. Этапы реализации

### Этап 0 — Фиксация среды и протокола (P0)

**Задачи:**
- [x] Проверить текущий код/зависимости проекта и зафиксировать baseline состояния Git.
- [x] Выбрать Python version и менеджер окружения; завести lockfile, `pyproject.toml`, lint/type/test configuration.
- [x] Создать конфиги путей без жёстких абсолютных путей, `.gitignore` для больших данных и outputs.
- [x] Согласовать названия datasets, единицы измерения, единый record schema, форматы прогнозов.
- [x] Зафиксировать protocol v1, random seeds, метод bootstrap и selection rule до обучения.

**Артефакты:** reproducible environment, `docs/benchmark_protocol.md`, минимальная CLI/CI skeleton.

**Приёмка:** установка и `pytest`/smoke запускаются без ручных правок путей; seed/config/code version отражаются в manifest run.

### Этап 1 — Аудит и индексирование данных (P0)

**Задачи:**
- [x] Потоково прочитать `train_synt_30_v7_anns.json` (~3.179 GB) и остальные JSON (`ijson`), без загрузки целого train JSON в память.
- [x] Проверить полноту `images`, `annotations`, корректность `image_id`, категории, RLE, `bbox`, `area`, диапазоны координат, существование PNG, возможные пустые и дублирующиеся маски.
- [x] Проверить глобальную/локальную уникальность `annotation.id`, `sequence_id`, `track_id` и временную согласованность track IDs.
- [x] Сравнить размеры, boxes и areas с фактически декодированными масками на репрезентативной выборке; полный контроль по ресурсам — отдельным проходом.
- [x] Сортировать синтетические кадры по числовому суффиксу; проверить, что кадры 0..29 действительно существуют для каждой sequence.
- [x] Для real построить mapping 342 записей к 120 физическим файлам, проверить противоречия в GT для повторов и согласовать canonical annotations.
- [x] Проверить наличие FPS, pixel size, provenance и непрерывности времени в источниках. Отсутствующие значения оставить `null`.
- [x] Сформировать индекс: `sequence → ordered frames → annotation offsets/objects`, чтобы не сканировать 3.18 GB на каждом вызове.
- [x] Создать статистический отчёт (instances/frame, размеры, плотность, длины, число пересечений, brightness distributions, распределение track lengths).
- [x] Зафиксировать checksum и версию данных и лицензионные ограничения.

**Артефакты:** `dataset_manifest.json`, индексированный annotation cache (например Parquet/SQLite), `audit_report.md` + графики, валидатор.

**Приёмка:** любое изображение/sequence читается по ID; нет незамеченных invalid refs; память при импорте ограничена; число реальных физических кадров в evaluation = 120 (или документированное меньшее число при исключении некорректных примеров).

### Этап 2 — Dataset API и визуальная проверка (P0)

**Задачи:**
- [x] Реализовать `get_frame`, `get_sequence`, итератор без полного preloading, фильтрацию по split, batch collate.
- [x] Возвращать grayscale array, boxes XYWH, lazy RLE masks, IDs, metadata, transform history.
- [x] Для временных моделей отдавать настоящие упорядоченные последовательности; не считать overlapping windows независимыми videos.
- [x] Реализовать augmentations с согласованным применением к кадру и instances; геометрические трансформации на все кадры последовательности единообразно при необходимости.
- [x] Реализовать инспектор: overlays masks, skeleton, boxes, цветные track IDs, ground truth trajectories.
- [x] Добавить тесты RLE roundtrip, точности больших IDs, обратного преобразования координат после resize/pad/crop.

**Артефакты:** `src/actin_tracking/data`, визуализации для случайных train/val/real samples, tests.

**Приёмка:** одно и то же GT корректно отображается во всех режимах; известные контрольные маски/IDs проходят roundtrip; никакие реальные кадры не подсчитываются повторно в evaluator.

### Этап 3 — Общие API модели и evaluator (P0)

**Задачи:**
- [ ] Зафиксировать абстракции `FramePrediction`, `SequencePrediction`, `Detector`, `Segmenter`, `Tracker`, `Measurement`.
- [ ] Реализовать сохранение предсказаний (JSONL/COCO RLE + Parquet tables) и версионирование схемы.
- [ ] Detection/instance segmentation evaluator с one-to-one matching, IoU thresholds, AP/F1 и категориями ошибок.
- [ ] Tracking evaluator (TrackEval или эквивалент) с чётким mapping frame IDs, обработкой empty frames и пропущенных предсказаний.
- [ ] Evaluation длины / скорости с физическими единицами и явной веткой `unit_unknown`.
- [ ] Реализовать uncertainty/summary report и runtime benchmark.
- [ ] Unit tests для TP/FP/FN, ID switches, edge cases, нулевых/пустых предсказаний.

**Артефакты:** единый evaluator + фиксированный synthetic smoke benchmark.

**Приёмка:** два метода, возвращающие одинаковые предсказания, получают точно одинаковые метрики; тесты с искусственно заданными ошибками дают ожидаемый ответ.

### Этап 4 — B0: Threshold + morphology + Hungarian (P0)

**Задачи:**
- [ ] Варианты background correction, smoothing, global/adaptive threshold.
- [ ] Connected components, morphology, фильтрация по площади/форме, skeletonization.
- [ ] Получить instances, boxes, approximate length, центры; явно описать проблему соединений/пересечений.
- [ ] Hungarian matching по distance/gating; max displacement, max missing, правила рождения/завершения треков.
- [ ] Подобрать небольшую сетку параметров по synthetic train/val без использования real test.
- [ ] Сохранить overlays и базовый error taxonomy.

**Артефакты:** `configs/models/b0*.yaml`, pipeline и оценка synthetic/real.

**Приёмка:** воспроизводимый CLI inference/evaluation; результат включает метрики и визуализации; отсутствует leakage from real eval.

### Этап 5 — B1: Ridge enhancement + Kalman/Hungarian (P0)

**Задачи:**
- [ ] Реализовать Frangi/Sato как опциональную предобработку с масштабами под толщину филаментов.
- [ ] Сравнить threshold/postprocessing при одинаковом tracking, затем Kalman + Hungarian при одинаковом segmentation.
- [ ] Провести ablation: полезность ridge filter vs дополнительная сложность.
- [ ] Разобрать ошибки на crossings, low contrast и crowded scenes.

**Артефакты:** конфиги `b1`, отчёт B0 vs B1.

**Приёмка:** можно выбрать B0/B1 одной опцией CLI, сравнение целиком воспроизводимо.

### Этап 6 — Оригинальный ATLAS как reference baseline (P1, можно вести параллельно B0/B1)

**Задачи:**
- [ ] Сверить опубликованный репозиторий, лицензию, версии Python/R/зависимостей, веса, входы и выходы.
- [ ] Запустить оригинальную реализацию на небольшом контрольном наборе с оригинальными pretrained weights, без изменения алгоритма.
- [ ] Проверить ожидаемые этапы статьи: YOLOv5 ensemble (5 models), merged boxes + edge handling/NMS/containment, crop U-Net, skeleton/branch rejection + centroid correction, DeepSORT, RMSD/velocity и MTS.
- [ ] Зафиксировать версии, настройки и ограничения оригинальной реализации; при проблемах совместимости документировать патчи отдельно.
- [ ] Написать adapter для нормализованного формата predictions/metrics, сохранив raw output оригинала.
- [ ] Оценить synthetic/real и записать долю отброшенных объектов, в том числе из-за пересечений/границ.

**Артефакты:** `atlas_official` wrapper, environment/runbook, raw+normalized outputs, reproduction log.

**Приёмка:** контрольный запуск на реальной и синтетической последовательности завершён либо документирован конкретный воспроизводимый blocker; для статуса «готово» blocker должен быть устранён. Результаты отмечены как `official pretrained`, а не как собственная обученная модель.

### Этап 7 — B2: U-Net semantic segmentation + instance extraction (P1)

**Задачи:**
- [ ] Создать training crops/frames с masks на синтетике; обучать U-Net с фиксированными seed, splits и augmentations.
- [ ] Рассмотреть imbalance-aware BCE+Dice; отделить semantic segmentation от восстановления instances.
- [ ] Для instance extraction реализовать connected components/watershed/skeleton-based separation и фиксировать failure cases на пересечениях.
- [ ] Использовать B0/B1 tracking в стандартном режиме; не добавлять oracle информацию в inference.
- [ ] Измерить synthetic→real domain gap; real fine-tuning только при допустимом независимом split.

**Артефакты:** checkpoints, model card, конфиги обучения/inference, отчёт ablation.

**Приёмка:** обученная модель воспроизводимо оценивается с общей системой; mask AP и tracking сравнимы с classical methods.

### Этап 8 — B3: Instance segmentation model + tracking (P1)

**Задачи:**
- [ ] Выбрать конкретную поддерживаемую версию YOLO-seg/другой instance segmenter и зафиксировать версию/лицензию.
- [ ] Конвертировать COCO RLE в пригодный формат train с проверкой геометрии тонких объектов; избегать некорректного упрощения полигонов.
- [ ] Обучить на synthetic train, проверить synthetic val и held-out real.
- [ ] Сравнить Hungarian, ByteTrack/BoT-SORT (при корректном интерфейсе и конфигурации для данного формата данных).
- [ ] Оценить чувствительность к разрешению, overlap/crossing, domain gap и runtime.

**Артефакты:** checkpoints/configs, tables B0–B3.

**Приёмка:** воспроизводимая instance-level inference и tracking с общим output schema.

### Этап 9 — Собственная модульная ATLAS reproduction (P1)

**Задачи:**
- [ ] Разложить оригинал на компоненты `detector`, `merge/postprocess`, `crop+UNet`, `skeleton/center`, `DeepSORT`, `measurements`, `MTS`.
- [ ] Воспроизвести paper-specific thresholds и логику с тестами для границ, containment, branches, centroid correction, track initiation.
- [ ] Реализовать контрольную проверку совпадения **промежуточных результатов** между official и reimplementation на маленькой фиксированной выборке.
- [ ] Разделить режимы `paper_default`, `adapted_to_dataset`; изменения параметров не выдавать за точную репликацию.
- [ ] При необходимости обучить отдельный локальный YOLOv5 ensemble и crop U-Net на имеющейся синтетике — обозначить как `atlas_retrained`, не путать с оригинальными весами.
- [ ] Подтвердить метод подсчёта длины и скорости (skeleton path, MSD/RMSD; особенности треков, калибровка) тестами на простых симулированных движениях.

**Артефакты:** `atlas_reimpl`, parity report, test cases, documented deviations.

**Приёмка:** модульный pipeline даёт результаты в пределах зафиксированных допусков относительно original на контрольном наборе; все отклонения объяснены и документированы. Точная parity может потребовать дополнительных зависимостей/настроек, это отдельная проверяемая цель.

### Этап 10 — Oracle studies, ablations и error analysis (P1)

**Задачи:**
- [ ] `GT masks + Hungarian/Kalman`: верхняя диагностическая оценка tracking при идеальной сегментации.
- [ ] `GT boxes + DeepSORT`: насколько ошибается association при идеальных detections.
- [ ] `predicted masks + GT association`: отделить errors масок от errors matching (только оценочный oracle).
- [ ] Ablation ATLAS: ensemble vs single detector; edge/containment/crossing handling on/off; corrected vs bbox center; DeepSORT vs simple association; MTS on/off.
- [ ] Cross-domain report: synthetic vs real, breakdown by SNR, density, filament length, intersections.
- [ ] Подготовить качественные panels: лучшие/худшие случаи и частые категории ошибок.

**Артефакты:** `experiments/ablations/`, отчёт bottlenecks, список гипотез улучшений.

**Приёмка:** объяснено, какую долю ошибок обусловливают segmentation/detection, association и измерения; результат подкреплён метриками и конкретными примерами.

### Этап 11 — Итоговый benchmark и удобство использования (P1)

**Задачи:**
- [ ] Реализовать устойчивый CLI и batch inference на произвольной директории/sequence manifest.
- [ ] Реализовать экспорт masks/tracks в COCO RLE + tracks table и rendered annotated video/images.
- [ ] Обеспечить сохранение полного run manifest, версии данных, хэшей weights, seed, hardware, runtime.
- [ ] Написать сравнение всех методов на неизменных splits (accuracy, speed, memory, coverage).
- [ ] Подготовить README с quickstart (tiny fixture), командами train/eval, схемами данных, FAQ и ограничениями.
- [ ] Добавить CI smoke tests (без большого датасета и GPU) и локальные опциональные GPU integration tests.

**Артефакты:** `reports/final_benchmark.md` или генерируемый отчёт, рабочий CLI, installation guide, тесты.

**Приёмка:** от чистого checkout до smoke inference/evaluation — по инструкции, без ручной правки исходного кода; итоговый отчёт пересобирается командой.

## 6. Зависимости между этапами и порядок работ

```text
Этап 0 (protocol/env)
  └─ Этап 1 (audit/index)
       └─ Этап 2 (Dataset API)
            └─ Этап 3 (Evaluator + contracts)
                 ├─ Этап 4 (B0) ─ Этап 5 (B1)
                 ├─ Этап 6 (ATLAS official)
                 ├─ Этап 7 (B2 U-Net)
                 └─ Этап 8 (B3 instance seg)
                      └─ Этап 9 (ATLAS reimplementation; использует также Этап 6)
                            └─ Этап 10 (oracle/ablation)
                                  └─ Этап 11 (final benchmark)
```

Этап 6 можно начинать параллельно ранним baseline после готовности минимального адаптера. Этап 9 зависит от понимания и запуска оригинала; Этап 10 можно частично выполнять раньше на B0/B1.

**Milestone M1 — готов фундамент:** этапы 0–3, dataset report + evaluator.  
**Milestone M2 — появились независимые baseline-метрики:** B0, B1 + сравнение synthetic/real.  
**Milestone M3 — оригинальный ATLAS запущен:** official pretrained и его результаты.  
**Milestone M4 — обучаемые бейзлайны:** B2, B3, checkpoints + сравнение.  
**Milestone M5 — репликация и исследование:** reimplementation, parity, oracle, ablations.  
**Milestone M6 — финальный продукт:** единый CLI, отчёт и документация.

*Оценки сроков намеренно не указаны: они сильно зависят от доступного GPU, качества track_id, формата исходных аннотаций и совместимости оригинального ATLAS.*

## 7. Пример целевого CLI (контракт, не существующие команды)

```bash
actin audit --dataset configs/datasets/v0.2.1.yaml --output outputs/audit_v021
actin index --dataset configs/datasets/v0.2.1.yaml
actin visualize --dataset synthetic_val --sequence-id 1074658952804 --output outputs/preview

actin infer --model b0 --config configs/models/b0.yaml --dataset synthetic_val --output outputs/b0_val
actin evaluate --predictions outputs/b0_val --dataset synthetic_val --output outputs/b0_eval

actin train --model b2 --config configs/models/b2.yaml
actin infer --model atlas_official --dataset real_eval --output outputs/atlas_real
actin benchmark --config configs/experiments/final_benchmark.yaml --output outputs/benchmark
actin report --runs outputs/benchmark --output outputs/benchmark/report.md
```

Параметр `--sequence-id` приведён для иллюстрации; ID требуется подтвердить в manifest. CLI должен уметь `--help`, `--dry-run`, понятные сообщения об ошибках, запись config snapshots.

## 8. Минимальная структура результатов одного эксперимента

```text
outputs/<run_id>/
├── config.resolved.yaml
├── manifest.json          # git sha, dataset hashes, seed, weights, hardware
├── predictions.jsonl      # COCO RLE + identifiers, schema version
├── tracks.parquet
├── metrics.json
├── per_sequence_metrics.csv
├── error_breakdown.csv
├── visualizations/
└── logs/
```

Большие файлы — не коммитить в Git. Для обучения отдельно сохранять weights/checkpoint references и журнал обучения. Никакие показатели качества не объявлять заранее: итоговые значения появятся только после реальных запусков.

## 9. Ключевые риски и действия

| Риск | Что делаем |
|---|---|
| Неполные/некорректные synthetic track IDs | аудит, тесты непрерывности, сегментировать только надежные sequences для tracking evaluation |
| У real нет GT trajectories | segmentation evaluation; отдельно получить GT track IDs для выбранных независимых клипов, если tracking на real — обязательный KPI |
| Три диапазона real могут быть не последовательными | запрет tracking evaluation, пока не подтверждены условия съемки и frame order |
| Малая real test выборка / tuning leakage | фиксированный split и отдельный exploratory режим; честно обозначать ограничения |
| 3.179 GB JSON / RLE overhead | stream parsing, one-time indexing, lazy decode, memory tests |
| Domain gap synthetic→real | отдельно оценивать domain-specific и cross-domain; optional adaptation только без test leakage |
| Филаменты пересекаются / накладываются | errors per crossing; oracle studies; мониторинг доли отброшенных объектов |
| ATLAS выбрасывает крайние/пересекающиеся объекты | сравнивать coverage и quality на raw и filtered populations |
| Расхождение собственной реплики и оригинала | контрольные outputs промежуточных stages + parity tests |
| Нет физической калибровки | не выдавать px/frame за µm/s; сохранять nullable metadata |
| Лицензии ATLAS, pretrained weights и исходного датасета | проверить до публикации/распространения/коммерческого применения |
| Зависимости Python/R/YOLOv5 устаревают | отдельное зафиксированное окружение для original и адаптер; не менять original молча |

## 10. Приоритеты для ближайшей итерации

**Первый PR (наиболее полезен):**

1. Зафиксировать окружение и схемы (`schemas.py`, manifests, benchmark protocol).
2. Потоковый dataset audit + полнота IDs + real deduplication + тесты на больших идентификаторах.
3. Минимальный loader для одного кадра и последовательности + визуальный overlay.
4. Тесты на three-frame real windows, synthetic ordering, RLE roundtrip.

**Второй PR:** evaluator + B0 + reproducible report для synthetic val и real unique frames.

**Третий PR:** B1 + ablation сравнение B0/B1; параллельно — isolated official ATLAS runbook.

### Критерии приёмки первого PR

- [ ] `actin audit` формирует manifest и summary без OOM на полном train JSON.
- [ ] `actin visualize` корректно показывает выбранные synthetic/real примеры.
- [ ] На synthetic sequences соблюдён порядок 0..29; известные большие IDs сохраняются побитово точно.
- [ ] Real evaluation использует уникальные кадры и не трактует перекрывающиеся трёхкадровые окна как независимые видео.
- [ ] Все обнаруженные проблемы данных отражены в отчёте, а не незаметно исправлены.

## 11. Открытые вопросы (не блокируют начало разработки)

- Есть ли достоверные `fps`, `µm/px` и источник этих значений для каждого видео?
- Что означает синтетический `track_id`: уникален ли он в пределах sequence, и есть ли полнота по всем объектам?
- Являются ли реальные диапазоны по 40 кадров непрерывными съемками и насколько они независимы?
- Можно ли использовать исходные реальные данные для обучения без потери чистого test set?
- Какие hardware/GPU доступны и какой бюджет по времени inference?
- Что важнее в конечном продукте: полная autoannotation всех экземпляров (включая crossings) или точность velocity/length measurements у подмножества надёжных филаментов?

## 12. Ссылки

1. Данные проекта: https://github.com/Knohhon/actin_tracking/blob/main/docs/data_overview.md
2. ATLAS paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12271436/
3. Official ATLAS implementation: https://github.com/obilogy/ATLAS
4. ATLAS/SAMY Zenodo: https://zenodo.org/records/15320358

---

**Итоговое определение успеха:** реализована система, в которой любой поддерживаемый метод можно обучить (если требуется), применить к одинаковым последовательностям, оценить одним протоколом, визуально разобрать ошибки и сравнить с оригинальным ATLAS. Результаты на синтетике и реальных данных не смешиваются, все ограничения GT и калибровки указаны явно, а запуск воспроизводим из версионированных конфигураций.
