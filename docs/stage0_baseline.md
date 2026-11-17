# Исходное состояние этапа 0

Проверено 2026-11-17. HEAD до реализации:
`3e5051e8eeb4066155a5d3f315311b2701a734c6` (`Upd: update docs`).

До начала: src/scripts/experiments содержали только .gitkeep; pyproject,
зависимостей, тестов и CLI не было. Python локально 3.12.0.
План `docs/ACTIN_TRACKING_IMPLEMENTATION_PLAN.md` был untracked;
в нём отмечены выполненные задачи этапа 0. Данные не изменялись.

Выбор окружения: Python 3.12, uv 0.6.17, uv.lock; runtime без сторонних
зависимостей, dev — pytest, Ruff, mypy. TOML вместо YAML позволяет
читать конфигурацию стандартной библиотекой.
