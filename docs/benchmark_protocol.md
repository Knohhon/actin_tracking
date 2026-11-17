# Benchmark protocol v1

Зафиксирован до обучения. Изменения правил требуют новой версии протокола;
метрики разных версий не объединять. Реализация evaluator — этап 3.

## Данные и единицы

- `synthetic_train`: обучение и настройка параметров только здесь. Внутреннее
  tuning-разбиение по sequence_id: SHA256 UTF-8 ID как целое modulo 5 == 0
  — tuning, остальные — fitting. Проверить отсутствие дубликатов до обучения.
- `synthetic_val`: фиксированная валидация, не использовать для подбора параметров.
- `real_eval`: уникальные физические PNG (file_name); 342 window-записи не
  являются 342 независимыми кадрами. Конфликты GT фиксировать при аудите;
  до их разрешения канонический GT не объявлять проверенным.
- Разметки real не являются разными независимыми выборками. Tracking на real
  запрещён до подтверждения непрерывности и GT identities. Fine-tuning на real
  меняет статус результатов на exploratory, если независимый test не выделен.
- Координаты исходного кадра; длина px, скорость px/frame. µm и секунды только
  с положительной калибровкой и provenance. Неизвестные значения — null.

## Seeds и выбор модели

Основной seed 20261117; для обучаемых методов дополнительные повторы
20261118, 20261119. Перед использованием NumPy/Torch явно инициализировать
их RNG и записать deterministic settings; один random.seed этого не заменяет.
Параметры/порог выбираются на synthetic_train tuning по macro instance F1
при mask IoU 0.5; tie-break: меньший runtime, затем лексикографический config ID.
Для обучения checkpoint выбирается тем же правилом, не по synthetic_val/real.
Финальный запуск сохраняет config, seeds, git SHA/dirty status, hashes данных
и weights, версии библиотек, hardware. Smoke пока не читает данные/weights.

## Matching и отчёт

Для фиксированного порога confidence 0.5: one-to-one matching отдельно на кадре,
максимальная кардинальность среди IoU >= 0.5, затем максимальная сумма IoU;
равенства разрешать по стабильным instance_id. Box и mask matching независимы.
Без совпадения GT — FN, предсказание — FP. Оба множества пусты: F1 null;
IoU/Dice усреднять только по matched pairs, обязательно вместе с FP/FN.
Для AP использовать score-ranked COCO evaluation с IoU 0.50:0.05:0.95,
maxDets=[1,10,1000], объявлять это отличие от COCO default в отчёте;
не обрезать плотные кадры молча. Категория одна: filament.

Публиковать detection precision/recall/F1, AP50/mAP, mask AP и matched IoU/Dice;
tracking HOTA/IDF1/IDSW/fragmentation только на проверенных GT sequences.
Длины/скорости оценивать лишь при согласованном GT и методе измерения.
Показывать micro по объектам и macro по sequences, N и coverage;
synthetic/real и raw/ATLAS-filtered populations строго отдельно.
Пустые/недоступные метрики — null с причиной, не 0.

95% CI: percentile cluster bootstrap 2000 повторов с seed 20261117,
resample независимых sequences с возвращением, сохранять все кадры кластера,
пересчитывать агрегат; quantiles 0.025/0.975 с линейной интерполяцией.
Real: CI по трём неподтверждённым диапазонам не строить.
Runtime: 10 warmup frames, затем полный фиксированный eval, три повтора;
ms/frame и throughput без загрузки модели, отдельно end-to-end и peak RAM/VRAM,
одинаковое оборудование, batch size и синхронизация GPU.

Пока аудит не завершён, dataset hashes, независимость splits и валидность GT
не подтверждены. Этот документ фиксирует правила, а не результаты проверки.
