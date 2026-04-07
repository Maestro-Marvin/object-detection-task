# Scene Understanding Pipeline

Конвейер для анализа сцен: по RGB-кадрам и маскам инстансов выбираются **опорные объекты** (по списку `id`), для каждого строятся кропы, далее VLM предсказывает **список предметов**, ассоциированных с опорным объектом. Отдельная VLM строит **детальные описания** найденных предметов (форма/материал/цвет/надписи + on/inside/near). Затем выполняется **локализация** этих лейблов на *исходных кадрах* с помощью **SAM3** (Ultralytics, text-prompt segmentation). Результаты оцениваются по ground truth с учётом синонимов и семантической близости.

## Структура проекта

```
.
├── config.py              # Пути, имена моделей, пороги
├── main.py                # Точка входа: пайплайн от кадров до метрик
├── requirements.txt
│
├── scenes/
│   └── scene2/
│       └── support_ids.json   # ID опорных объектов (редактируется без правки кода)
│
├── support_objects/      # Опорные объекты и отбор кропов
│   ├── select_support_object.py   # Выбор опорных по маске и списку SUPPORT_OBJECT_IDS
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
│   ├── scene_understanding.py  # Задача: вернуть JSON-массив лейблов предметов
│   ├── item_detailer.py   # Детальное текстовое описание предметов + on/inside/near
│   ├── gt_refinement.py   # Уточнение GT: вернуть JSON-массив лейблов
│   └── crop_selector.py   # Сравнение двух кропов, выбор лучшего (A/B)
│
├── evaluate/
│   ├── evaluator.py       # Сопоставление предсказаний с GT: синонимы → эмбеддинги
│   ├── embedding_matcher.py  # Семантическое сходство (Qwen3-Embedding), жадный матчинг
│   └── calculate_metrics.py  # Precision, Recall, F1 по TP/FP/FN
│
├── sam3/
|   └── weights/
|       |── sam3.pt
|       |── sam3.safetensors
│   ├── sam3_localization.py  # Инференс SAM3 и получение масок по text prompt
│   └── sam3_rendering.py     # Отрисовка и сохранение оверлеев/масок
│
└── results/               # Выходные JSON (настраиваются в config.py)
    ├── temp_gt.json       # Кандидаты GT по объектам
    ├── selected_crops.json # Кэш отобранных кропов (по obj_id)
    ├── predictions.json   # Предсказания: id -> [label, ...]
    ├── detailed_predictions.json # Детальные текстовые описания по id
    ├── ground_truth.json  # Уточнённый GT: id -> [label, ...]
    ├── report.json        # Детальный отчёт: tp/fp/fn по каждому id
    └── metrics.json       # Итоговые метрики
    
└── localization/          # Выход локализации SAM3
    ├── overlays/          # Оверлеи с масками на исходных кадрах
    └── masks/             # Бинарные маски (union) по каждому label/кадру
```

## Входные данные

- **Сценарии** в `scenes/` (пути задаются в `config.py`):
  - `scenes/rgb/` — RGB-кадры (jpg/jpeg)
  - `scenes/gt_instance_iphone/render_instance_npy/` — маски инстансов (`.npy`, имя кадра без расширения)
  - `scenes/gt_categories.json` — описания объектов: `dataset.samples[].object_id`, `labels.image_attributes.synonyms` (список синонимов для матчинга и оценки)
- **Список опорных объектов**:
  - `scenes/scene2/support_ids.json` — JSON-массив `id` опорных объектов, например: `[15, 16, 79, ...]`

## Пайплайн (кратко)

Модели запускаются **последовательно**, чтобы не держать несколько тяжёлых VLM одновременно в VRAM.

1. **Кропы и кандидаты GT**: обход кадров, выбор опорных объектов по списку `support_ids.json`, сохранение кропов в `crops/<obj_id>/`, построение `results/temp_gt.json`.
2. **Отбор лучших кропов**: `CropSelectorVLM` выбирает до `MAX_CROPS_PER_REQUEST` кропов на объект, кэш в `results/selected_crops.json`.
3. **Предсказание лейблов**: `SceneUnderstandingVLM` возвращает **валидный JSON-массив строк** (лейблы предметов), сохраняется в `results/predictions.json`.
4. **Детализация**: `ItemDetailerVLM` получает выбранные кропы + предсказанные лейблы и возвращает детальные атрибуты/описания. Сохраняется в `results/detailed_predictions.json`.
5. **Локализация (SAM3)**: `SAM3Localizer` сегментирует каждый `label` на *исходных изображениях* из `FRAMES_DIR` для выбранных кадров (`selected_crops.json`). Результаты сохраняются в `localization/overlays/` и `localization/masks/`.
6. **Уточнение GT**: `GTRefinementVLM` (опционально) возвращает JSON-массив лейблов для GT и пишет в `results/ground_truth.json`.
7. **Оценка**: `evaluate/evaluator.py` сопоставляет списки лейблов, сначала по словарю синонимов, затем по эмбеддингам; формирует `results/report.json` и `results/metrics.json`.

## Модели (config.py)

| Переменная | Назначение |
|------------|------------|
| `TASK_MODEL_NAME` | Основная VLM для scene understanding (например, `nvidia/Cosmos-Reason2-8B` или `Qwen/Qwen3-VL-8B-Instruct`) |
| `SELECTOR_MODEL_NAME` | VLM для турнирного выбора лучших кропов |
| `DETAIL_MODEL_NAME` | VLM для детального описания предметов |
| `EMBED_MODEL_NAME` | Модель эмбеддингов для матчинга при оценке (Qwen3-Embedding-8B) |
| `SAM3_MODEL_PATH` | Путь к весам SAM3 (`.pt`) |
| `SAM3_CONF`, `SAM3_HALF` | Параметры инференса SAM3 |
| `SAM3_SAVE_BINARY_MASKS` | Сохранять ли бинарные маски (union) |

Для Cosmos-Reason2 в системный промпт добавляется формат ответа с тегами `<think>` и `<answer>`; из вывода используется только содержимое `<answer>`.

## Запуск

Требуется GPU с достаточным объёмом памяти (vLLM для VLM и эмбеддингов).

```bash
pip install -r requirements.txt
python main.py
```

Перед запуском проверьте в `config.py` пути к данным (`DATA_ROOT`, `FRAMES_DIR`, `MASKS_DIR`, `DESC_PATH`) и список `scenes/scene2/support_ids.json`.

Скачайте веса SAM3 с помощтю команды

```bash
hf download 1038lab/sam3 --local-dir ./sam3/weights
```

## Зависимости

- Python 3.x
- `vllm`, `transformers`, `qwen_vl_utils` — инференс VLM и обработка мультимодальных запросов
- `numpy`, `opencv-python`, `PIL` — загрузка и сохранение изображений и масок
- `torch` — эмбеддинги и метрики (через vLLM/embedding-модель)

## Формат отчёта и метрик

- **report.json**: для каждого `id` — списки `tp` (пары "pred - gt"), `fp`, `fn`.
- **metrics.json**: общие Precision, Recall, F1 и суммарные Total_TP, Total_FP, Total_FN.

Матчинг при оценке: сначала точное совпадение по словарю синонимов (из описаний), для оставшихся — жадное сопоставление по косинусной близости эмбеддингов с порогом `SIMILARITY_THRESHOLD`.

## Форматы выходных файлов (results/)

### predictions.json
Словарь `id -> список лейблов`:

```json
{
  "id_79": ["laptop", "earphones", "pillow"]
}
```

### detailed_predictions.json
Словарь `id -> список объектов` с детальными атрибутами по каждому label (примерный формат):

```json
{
  "id_79": [
    {
      "label": "laptop",
      "relation": "on",
      "shape": "rectangular",
      "material": "metallic",
      "color": "silver",
      "text_markings": "apple logo",
      "confidence": "high"
    }
  ]
}
```

## Локализация (SAM3)

Локализация выполняется на **исходных кадрах** (`FRAMES_DIR`), а не на кропах. Для каждого `label` из `predictions.json` модель SAM3 получает text-prompt (сам `label`) и выдаёт маску.

- **Оверлеи**: `localization/overlays/<frame>__id_<obj_id>__<label>.jpg`
- **Маски** (union): `localization/masks/<frame>__id_<obj_id>__<label>.png`

### ground_truth.json
Словарь `id -> список лейблов` (после уточнения `GTRefinementVLM`).
