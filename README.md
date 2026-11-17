# Actin tracking

Фундамент воспроизводимого сравнения методов разметки актиновых филаментов.
Реализован этап 0 плана: окружение, контракт данных, протокол и CLI smoke.
Аудит, модели и вычисление метрик пока не реализованы.

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

Версии зависимостей зафиксированы в `uv.lock`. Runtime пока использует только
стандартную библиотеку; зависимости обработки изображений добавятся на этапе 1.

- [План реализации](docs/ACTIN_TRACKING_IMPLEMENTATION_PLAN.md)
- [Протокол benchmark v1](docs/benchmark_protocol.md)
- [Контракт данных](docs/data_contract.md)
- [Исходные данные и ограничения](docs/data_overview.md)

Исходные данные неизменяемы и исключены из Git; результаты пишутся в `outputs/`.
Физическая калибровка неизвестна; единицы пока только px и px/frame.
