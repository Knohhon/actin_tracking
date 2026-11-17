# Actin tracking

Фундамент воспроизводимого сравнения методов разметки актиновых филаментов.
Реализованы окружение, контракт данных, протокол, CLI smoke и потоковый
аудит/индексирование. Модели и вычисление benchmark-метрик пока не реализованы.

## Установка и проверка

Нужны Python 3.12 и uv 0.6.17 (`python -m pip install uv==0.6.17`).
Из корня репозитория:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run actin --help
uv run actin smoke --config configs/datasets/v0.2.1.toml --dry-run
uv run actin smoke --config configs/datasets/v0.2.1.toml --output outputs/first_smoke
```

Smoke не требует большого датасета или GPU. Создаёт `manifest.json` и
`config.resolved.json`; существующий каталог запуска не перезаписывает.
`--dry-run` проверяет конфиг без записи файлов. Пути конфигурации разрешаются
относительно TOML-файла; `--data-root /path/to/v0.2.1` переопределяет корень
относительно текущей директории. Seed по умолчанию — 20261117.

Версии зависимостей зафиксированы в `uv.lock`. Аудит использует ijson, pycocotools, NumPy и Pillow; установка производится
по тому же lockfile.

- [План реализации](docs/ACTIN_TRACKING_IMPLEMENTATION_PLAN.md)
- [Протокол benchmark v1](docs/benchmark_protocol.md)
- [Контракт данных](docs/data_contract.md)
- [Аудит и SQLite-индекс](docs/data_audit.md)
- [Результат локальной проверки](docs/stage1_results.md)
- [Исходные данные и ограничения](docs/data_overview.md)

Исходные данные неизменяемы и исключены из Git; результаты пишутся в `outputs/`.
Физическая калибровка неизвестна; единицы пока только px и px/frame.

## Аудит и индексирование (этап 1)

```bash
uv run actin audit --config configs/datasets/v0.2.1.toml --output outputs/audit_v021
# Быстрая проверка validation и обоих real JSON:
uv run actin audit --config configs/datasets/v0.2.1.toml --subset synthetic_val --subset real_eval --output outputs/audit_val
# Декодировать каждую маску вместо детерминированной выборки из 100:
uv run actin audit --config configs/datasets/v0.2.1.toml --subset real_eval --full-masks --output outputs/audit_real_full
```

`actin index` — синоним `actin audit` (индекс всегда проходит проверки).
`--dataset` — синоним `--config`. `--subset` можно повторять, без него проверяются
все выборки. Все JSON читаются потоково, PNG хэшируются целиком; полный train
требует времени и нескольких GB свободного места для SQLite. Исходники не меняются.
Выходной каталог должен быть новым. Код завершения: 0 — проблем не найдено,
1 — аудит завершён с замечаниями, 2 — ошибка запуска/входного файла.

Результаты: `dataset_manifest.json`, `audit_report.md`, `*_areas.svg`,
`*.sqlite`, `config.resolved.json`. Все замечания доступны через SQLite:

```sql
SELECT code, record_type, record_id, detail FROM issues ORDER BY code;
```

Индекс предоставляет доступ к исходной разметке без чтения большого JSON:

```python
from pathlib import Path
from actin_tracking.data.indexing import DatasetIndex

with DatasetIndex(Path("outputs/audit_v021/synthetic_val.sqlite")) as index:
    frame = index.get_frame("<image_id>")
    for frame in index.get_sequence("<sequence_id>"):
        print(frame["image"]["file_name"], len(frame["annotations"]))
```

`iter_unique_frames()` даёт по одной канонической записи на физический PNG;
конфликтующие и недоступные/невалидные PNG исключены. По ID можно прочитать
и проблемную исходную запись. IDs в SQL TEXT; исходные payload JSON сохраняют
точные Python integers. Дубликаты масок проверяются по точному RLE представлению.
Статистика описывает исходные записи; real окна повторяют физические кадры.
Декодированная геометрия проверяется по SHA256-ID sample либо `--full-masks`.
Диагностическая длина — число пикселей skeleton после Zhang–Suen thinning
на выборке масок. Для трёх уникальных кадров считаются пары масок с положительным
IoU: это overlap, а не подтверждённое пересечение осевых линий. Эти величины
не являются GT физических измерений. Калибровка остаётся неизвестной.

Под WSL для большого импорта удобно писать SQLite на Linux-диск, например
`--output /tmp/audit_v021`, затем перенести каталог артефактов в `outputs/`.
