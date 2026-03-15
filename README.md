# Scene Understanding Pipeline

Конвейер для анализа сцен: по RGB-кадрам и маскам инстансов определяются опорные объекты (стол, кровать, полка и т.д.), для каждого строятся кропы, а Vision-Language модель (VLM) предсказывает предметы, расположенные **на** / **внутри** / **рядом** с опорным объектом. Результаты оцениваются по ground truth с учётом синонимов и семантической близости.

## Структура проекта

```
.
├── config.py              # Пути, имена моделей, пороги
├── main.py                # Точка входа: пайплайн от кадров до метрик
├── requirements.txt
│
├── support_objects/      # Опорные объекты и отбор кропов
│   ├── select_support_object.py   # Выбор опорных по маске и семантике (SUPPORT_KEYWORDS)
│   ├── support_object_utils.py   # expand_bbox и др.
│   └── select_best_crops.py      # Турнирное прореживание кропов через VLM (до ≤5 на объект)
│
├── utils/
│   ├── data_loader.py     # Загрузка описаний (gt_categories.json), кадров и масок
│   ├── cropper.py         # Вырезка кропа по bbox, маскирование других опорных
│   ├── aggregator.py      # collect_crops_by_object, save_result
│   └── gt_builder.py      # Построение кандидатов GT по маскам (частота по кадрам)
│
├── vlm/                   # Vision-Language модели (vLLM + Qwen3-VL/Cosmos)
│   ├── base.py            # VLMClient: чат-шаблон, <think>/<answer> парсинг для Cosmos
│   ├── scene_understanding.py  # Задача: перечислить предметы on/inside/near объект
│   ├── gt_refinement.py   # Уточнение по списку кандидатов (опционально)
│   └── crop_selector.py   # Сравнение двух кропов, выбор лучшего (A/B)
│
├── evaluate/
│   ├── evaluator.py       # Сопоставление предсказаний с GT: синонимы → эмбеддинги
│   ├── embedding_matcher.py  # Семантическое сходство (Qwen3-Embedding), жадный матчинг
│   └── calculate_metrics.py  # Precision, Recall, F1 по TP/FP/FN
│
└── results/               # Выходные JSON (настраиваются в config.py)
    ├── temp_gt.json       # Кандидаты GT по объектам
    ├── selected_crops.json # Кэш отобранных кропов (по obj_id)
    ├── predictions.json   # Предсказания модели по объектам
    ├── ground_truth.json  # Уточнённый GT (если используется refiner)
    ├── report.json        # Детальный отчёт: tp/fp/fn по каждому id
    └── metrics.json       # Итоговые метрики
```

## Входные данные

- **Сценарии** в `scenes/` (пути задаются в `config.py`):
  - `scenes/rgb/` — RGB-кадры (jpg/jpeg)
  - `scenes/gt_instance_iphone/render_instance_npy/` — маски инстансов (`.npy`, имя кадра без расширения)
  - `scenes/gt_categories.json` — описания объектов: `dataset.samples[].object_id`, `labels.image_attributes.synonyms` (список синонимов для матчинга и оценки)

## Пайплайн (кратко)

1. **Загрузка** описаний и (опционально) обход кадров: выбор опорных объектов по маске и ключевым словам, сохранение кропов в `crops/<obj_id>/`, построение `temp_gt` и сохранение в `results/temp_gt.json`.
2. **Кропы на объект**: для каждого `obj_id` — список путей к кропам. Опционально: турнирный отбор через `CropSelectorVLM` до ≤ `MAX_CROPS_PER_REQUEST` с кэшем в `results/selected_crops.json`.
3. **Предсказание**: для каждого объекта вызывается VLM задачи (`SceneUnderstandingVLM`): по выбранным кропам модель возвращает строку вида `[item1, item2] on <object> id=<id>; ...` или `none`. Опционально: уточнение через `GTRefinementVLM` с кандидатами из `temp_gt`.
4. **Оценка**: парсинг предсказаний и GT, сопоставление по словарю синонимов, затем по эмбеддингам; подсчёт TP/FP/FN, вывод `report.json` и `metrics.json`.

## Модели (config.py)

| Переменная | Назначение |
|------------|------------|
| `TASK_MODEL_NAME` | Основная VLM для scene understanding (например, `nvidia/Cosmos-Reason2-8B` или `Qwen/Qwen3-VL-8B-Instruct`) |
| `SELECTOR_MODEL_NAME` | VLM для турнирного выбора лучших кропов |
| `EMBED_MODEL_NAME` | Модель эмбеддингов для матчинга при оценке (Qwen3-Embedding-8B) |

Для Cosmos-Reason2 в системный промпт добавляется формат ответа с тегами `<think>` и `<answer>`; из вывода используется только содержимое `<answer>`.

## Запуск

Требуется GPU с достаточным объёмом памяти (vLLM для VLM и эмбеддингов).

```bash
pip install -r requirements.txt
python main.py
```

Перед запуском проверьте в `config.py` пути к данным (`DATA_ROOT`, `FRAMES_DIR`, `MASKS_DIR`, `DESC_PATH`) и при необходимости раскомментируйте в `main.py` блок построения кропов и `temp_gt`, если вы начинаете с сырых кадров.

## Зависимости

- Python 3.x
- `vllm`, `transformers`, `qwen_vl_utils` — инференс VLM и обработка мультимодальных запросов
- `numpy`, `opencv-python`, `PIL` — загрузка и сохранение изображений и масок
- `torch` — эмбеддинги и метрики (через vLLM/embedding-модель)

## Формат отчёта и метрик

- **report.json**: для каждого `id` — списки `tp` (пары "pred - gt"), `fp`, `fn`.
- **metrics.json**: общие Precision, Recall, F1 и суммарные Total_TP, Total_FP, Total_FN.

Матчинг при оценке: сначала точное совпадение по словарю синонимов (из описаний), для оставшихся — жадное сопоставление по косинусной близости эмбеддингов с порогом `SIMILARITY_THRESHOLD`.
