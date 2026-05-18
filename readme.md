# Scene Understanding Pipeline

Конвейер для анализа сцен: по RGB-кадрам и маскам инстансов выбираются **опорные объекты** (по списку `id`), для каждого строятся кропы, VLM отбирает лучшие виды, предсказывает **список предметов** на/рядом с опорным объектом и уточняет **ground truth** по кандидатам из масок. Итог сравнивается с предсказаниями: сначала по словарю синонимов, затем по семантической близости эмбеддингов.

## Структура проекта

```
.
├── config.py                 # Пути, имена моделей, пороги
├── main.py                   # Точка входа: полный пайплайн
├── requirements.txt
│
├── scenes/                   # Данные по сценам (scene1 … scene4)
│   └── scene1/
│       ├── support_ids.json      # ID опорных объектов
│       ├── gt_categories.json    # Синонимы и описания объектов
│       ├── rgb/                  # RGB-кадры
│       └── gt_instance_iphone/
│           └── render_instance_npy/  # Маски инстансов (.npy)
│
├── pipeline/                 # Стадии конвейера
│   ├── base.py
│   ├── temp_gt.py            # Кропы + кандидаты GT из масок
│   ├── select_crops.py       # Турнирный отбор кропов через VLM
│   ├── scene_understanding.py  # Предсказание лейблов предметов
│   ├── gt_refinement.py      # Уточнение GT через VLM
│   └── evaluation.py         # Метрики pred vs GT
│
├── support_objects/
│   ├── select_support_object.py   # Выбор опорных по маске и support_ids
│   ├── support_object_utils.py    # expand_bbox и др.
│   └── select_best_crops.py       # Турнирное прореживание кропов
│
├── utils/
│   ├── data_loader.py        # Описания, кадры, маски
│   ├── cropper.py            # Вырезка кропа, маскирование других опорных
│   ├── aggregator.py         # collect_crops_by_object, save_result
│   ├── gt_builder.py         # Кандидаты GT по частоте на кадрах
│   └── prediction_parser.py  # Парсинг JSON-ответов VLM
│
├── vlm/                      # Vision-Language модели (vLLM + Qwen3-VL)
│   ├── base.py               # SharedVLMEngine, VLMClient
│   ├── crop_selector.py      # Сравнение двух кропов (A/B)
│   ├── scene_understanding.py
│   └── gt_refinement.py
│
├── evaluate/
│   ├── evaluator.py          # Синонимы → эмбеддинги
│   ├── embedding_matcher.py  # Qwen3-Embedding, жадный матчинг
│   └── calculate_metrics.py  # Micro/Macro Precision, Recall, F1
│
├── crops/                    # Кропы по объектам: crops/<obj_id>/<frame>.jpg
└── results/                  # Выходные JSON (пути в config.py)
    ├── temp_gt.json
    ├── selected_crops.json
    ├── predictions.json
    ├── ground_truth.json
    ├── report.json
    └── metrics.json
```

## Входные данные

Сцена задаётся через `DATA_ROOT` в `config.py` (по умолчанию `scenes/scene1`). Для каждой сцены нужны:

| Путь (относительно `DATA_ROOT`) | Содержимое |
|---------------------------------|------------|
| `rgb/` | RGB-кадры (`.jpg`, `.jpeg`) |
| `gt_instance_iphone/render_instance_npy/` | Маски инстансов (`.npy`, имя = имя кадра без расширения) |
| `gt_categories.json` | Описания: `dataset.samples[].object_id`, `labels.image_attributes.synonyms` |
| `support_ids.json` | JSON-массив `id` опорных объектов, например `[127, 15, 16]` |

В репозитории подготовлены сцены `scene1`–`scene4`; для другой сцены достаточно сменить `DATA_ROOT`.

## Пайплайн

`main.py` запускает пять стадий. VLM-стадии используют **один** экземпляр `SharedVLMEngine` (`TASK_MODEL_NAME`), чтобы не загружать модель в VRAM несколько раз.

```
TempGt → SelectCrops → SceneUnderstanding → GtRefinement → Evaluation
         └─ SharedVLMEngine ─────────────┘
```

1. **TempGt** (`TempGtStage`): обход кадров, выбор опорных по `support_ids.json`, сохранение кропов в `crops/<obj_id>/`, построение кандидатов GT в `results/temp_gt.json` (объекты внутри расширенного bbox опорного, встречающиеся на ≥ `FRAMES_SHARE` кадров).
2. **SelectCrops** (`SelectCropsStage`): турнирный отбор лучших кропов через `CropSelectorVLM` до `TOURNAMENT_TARGET_CROPS` (4) на объект; кэш в `results/selected_crops.json` (пропуск уже обработанных `obj_id`).
3. **SceneUnderstanding** (`SceneUnderstandingStage`): по отобранным кропам VLM возвращает JSON-массив строк-лейблов → `results/predictions.json`.
4. **GtRefinement** (`GtRefinementStage`): VLM уточняет список лейблов GT по кропам и кандидатам из `temp_gt.json` → `results/ground_truth.json`.
5. **Evaluation** (`EvaluationStage`): сопоставление `predictions.json` и `ground_truth.json` → `results/report.json`, `results/metrics.json`.

## Модели и параметры (`config.py`)

| Переменная | Назначение |
|------------|------------|
| `TASK_MODEL_NAME` | VLM для shared-движка (scene understanding, GT refinement) |
| `SELECTOR_MODEL_NAME` | Модель для `CropSelectorVLM` (должна совпадать с `TASK_MODEL_NAME`, если используется общий `SharedVLMEngine`) |
| `EMBED_MODEL_NAME` | Эмбеддинги для оценки (`Qwen/Qwen3-Embedding-8B`) |
| `MAX_CROPS_PER_REQUEST` | Лимит изображений в одном запросе к VLM (5) |
| `TOURNAMENT_TARGET_CROPS` | Целевое число кропов после турнира (4) |
| `FRAMES_SHARE` | Доля кадров для попадания объекта в `temp_gt` (0.5) |
| `SIMILARITY_THRESHOLD` | Порог косинусной близости эмбеддингов при оценке (0.65) |
| `MIN_BBOX_RATIO` | Минимальная доля площади bbox опорного на кадре |
| `PADDING_RATIO_GT` / `PADDING_RATIO_MODEL` | Расширение bbox для GT и для кропов |

## Запуск

Требуется GPU с достаточным объёмом памяти (vLLM для VLM; при оценке дополнительно загружается embedding-модель).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Перед запуском проверьте в `config.py`:

- `DATA_ROOT` — нужная сцена;
- пути `FRAMES_DIR`, `MASKS_DIR`, `DESC_PATH`, `SUPPORT_IDS_PATH`;
- имена моделей и `gpu_memory_utilization` в `main.py` (по умолчанию `0.5`).

Повторный запуск **SelectCrops** использует кэш: объекты, уже записанные в `selected_crops.json`, пропускаются. Чтобы пересчитать отбор, удалите соответствующие ключи или весь файл.

## Зависимости

- Python 3.11+
- `vllm`, `transformers`, `qwen_vl_utils` — инференс VLM
- `numpy`, `opencv-python`, `PIL` — кадры, маски, кропы
- `torch` — эмбеддинги при оценке

## Форматы выходных файлов

### temp_gt.json

Кандидаты GT из геометрии масок (до уточнения VLM):

```json
{
  "127": ["cup", "phone", "book"]
}
```

Ключи — числовые `object_id` опорных объектов.

### selected_crops.json

Кэш путей к отобранным кропам:

```json
{
  "127": ["crops/127/frame_001410.jpg", "crops/127/frame_001820.jpg"]
}
```

### predictions.json / ground_truth.json

Словарь `id_<obj_id> → список лейблов` (нижний регистр):

```json
{
  "id_127": ["cup", "phone", "book"]
}
```

### report.json

Список записей по каждому `id`:

```json
[
  {
    "id": "id_127",
    "tp": ["cup - cup", "phone - mobile phone"],
    "fp": ["pen"],
    "fn": ["notebook"]
  }
]
```

### metrics.json

Сводные метрики (micro и macro):

```json
{
  "Micro_Precision": 0.75,
  "Micro_Recall": 0.6,
  "Micro_F1": 0.6667,
  "Macro_Precision": 0.8,
  "Macro_Recall": 0.65,
  "Macro_F1": 0.71
}
```

## Оценка качества

`Evaluator` для каждого опорного объекта:

1. Парсит предсказания и GT как списки строк (или извлекает элементы из формата `[item1, item2]`).
2. Сопоставляет пары сначала по словарю синонимов из `gt_categories.json`.
3. Оставшиеся элементы матчит жадно по эмбеддингам (`EmbeddingMatcher`, порог `SIMILARITY_THRESHOLD`).

Итог: списки TP (пары `"pred - gt"`), FP и FN; по ним считаются micro/macro Precision, Recall и F1.
