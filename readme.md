# Пайплайн понимания сцены и локализации предметов

End-to-end скрипт **`main.py`** обрабатывает последовательность RGB-кадров одной сцены: находит **опорные объекты** (поверхности/мебель), предсказывает **предметы**, связанные с каждой опорой, строит **детальные описания**, затем **сегментирует** предметы на исходных кадрах через **SAM3** (Ultralytics) с опциональным выбором экземпляра маски второй VLM.

Один общий движок **vLLM** (`SharedVLMEngine`) используется всеми шагами на Qwen3-VL; после работы вызывается `shutdown()`.

## Структура проекта

```
.
├── config.py                 # Пути, имена моделей, параметры SAM3 и отбора кадров
├── main.py                   # Точка входа: 4 этапа пайплайна
├── requirements.txt
├── scenes/
│   └── scene2/
│       └── rgb/              # Входные кадры (.jpg / .jpeg); путь задаётся FRAMES_DIR
├── support_objects/
│   ├── tournament.py         # Протокол A/B и один раунд турнира
│   └── select_best_crops.py # До MAX_CROPS_PER_REQUEST кадров на группу опоры
├── utils/
│   └── aggregator.py        # save_result → JSON на диск
├── vlm/
│   ├── base.py              # SharedVLMEngine, VLMClient (Qwen-VL + vLLM)
│   ├── frame_consolidator.py # Этап 1: stride, группировка по опоре, турнир (+ внутренние _Support/_Frame селекторы)
│   ├── scene_understanding.py
│   └── item_detailer.py
├── sam3/
│   ├── weights/             # sam3.pt (скачать отдельно)
│   ├── sam3_localization.py
│   ├── sam3_rendering.py
│   ├── mask_chooser_vlm.py  # VLM выбора маски при нескольких кандидатах SAM3
│   └── localization_runner.py
├── results/                 # Создаётся при persist=True
└── localization/            # Оверлеи и маски SAM3
```

## Этапы пайплайна

1. **Консолидация кадров** (`FrameConsolidator.consolidate`)
   - Прореживание: каждый `FRAME_STRIDE`-й кадр.
   - По каждому кадру VLM классифицирует доминирующую опору (JSON `present` / `label`).
   - Кадры группируются по метке опоры; внутри группы турнир (`FrameSelectorVLM` + `select_best_crops_tournament`) оставляет не больше **`MAX_CROPS_PER_REQUEST`** кадров.
   - При `persist=True` пишутся `frames_by_support_raw.json`, `frames_by_support.json`, при непустом наборе опор — `selected_crops.json`.

2. **Список предметов по опоре** (`SceneUnderstandingVLM.predict_associated_items`)
   - На выбранных кадрах для каждой опоры — JSON-массив строковых лейблов (англ., родовые категории).
   - Сохранение: **`results/predictions.json`**. Ключи верхнего уровня — **названия опор** (как после группировки), не числовые id.

3. **Детальные описания** (`ItemDetailerVLM.predict_detailed_descriptions`)
   - Структурированные поля (форма, материал, цвет, relation on/inside/near и т.д.).
   - Сохранение: **`results/detailed_predictions.json`**; ключи совпадают с именами опор.

4. **Локализация SAM3** (`SAM3LocalizationRunner.localize_all`)
   - Для каждой опоры и каждого предсказанного предмета — text-prompt сегментация на **полноразмерных** кадрах из `FRAMES_DIR` (имя файла берётся из путей выбранных кропов).
   - Корень вывода: **`localization/<sanitize(опора)>/<sanitize(label)>/`** — подпапки `overlays/`, при `SAM3_SAVE_BINARY_MASKS` ещё `masks/`.
   - Если SAM3 возвращает несколько масок, top-K кандидатов показываются **`SAM3MaskChooserVLM`**; изображения кандидатов пишутся во **временный каталог** и удаляются после выбора (в дерево `localization/` промежуточные кандидаты не сохраняются).

## Входные данные

- Каталог кадров: **`config.FRAMES_DIR`** (по умолчанию `scenes/scene2/rgb`), файлы с расширениями `.jpg` / `.jpeg`, порядок по имени.

Пути смены сцены — правка **`DATA_ROOT`** / **`FRAMES_DIR`** в `config.py`.

## Выходные артефакты

| Путь | Содержание |
|------|------------|
| `results/frames_by_support_raw.json` | Группировка имён файлов по опоре до турнира |
| `results/frames_by_support.json` | После турнира |
| `results/selected_crops.json` | Выбранные кадры по ключам-именам опор |
| `results/predictions.json` | `{ "<опора>": ["item", ...], ... }` |
| `results/detailed_predictions.json` | `{ "<опора>": [ { "label", "relation", ... }, ... ], ... }` |
| `localization/...` | Оверлеи (и опционально бинарные маски) SAM3 |

## Конфигурация (`config.py`)

| Переменная | Назначение |
|------------|------------|
| `DATA_ROOT`, `FRAMES_DIR` | Корень сцены и RGB |
| `TASK_MODEL_NAME` | Основная VLM (группировка опор в консолидаторе совпадает по имени с shared engine в `main`) |
| `SELECTOR_MODEL_NAME` | Модель турнирного сравнения кадров |
| `DETAIL_MODEL_NAME` | Модель детализации и маски-чузера SAM3 |
| `MAX_CROPS_PER_REQUEST` | Максимум кадров на опору после турнира (и лимит изображений в запросе к VLM) |
| `FRAME_STRIDE` | Шаг прореживания исходной последовательности |
| `SAM3_MODEL_PATH`, `SAM3_CONF`, `SAM3_HALF` | Веса и параметры Ultralytics SAM3 |
| `SAM3_AGENT_TOPK` | Сколько масок-кандидатов показывать VLM при конфликте |
| `SAM3_SAVE_BINARY_MASKS` | Сохранять ли union-маски |

Пути к JSON в `results/` задаются константами `PRED_JSON`, `DETAILED_PRED_JSON`, `SELECTED_CROPS`, `FRAMES_BY_SUPPORT_*`.

## Запуск

Нужна среда с GPU и установленными зависимостями (в проекте типичен виртуальный env `.venv`).

```bash
pip install -r requirements.txt
python main.py
```

Веса SAM3 (пример загрузки через Hugging Face CLI):

```bash
hf download 1038lab/sam3 --local-dir ./sam3/weights
```

Ожидается файл весов по пути из `SAM3_MODEL_PATH` (например `sam3/weights/sam3.pt`).

## Зависимости (ключевые)

- **vLLM**, **transformers**, **qwen-vl-utils** — инференс Qwen-VL.
- **torch**, **ultralytics** — SAM3.
- **opencv-python**, **Pillow**, **numpy** — изображения и постобработка.

Полный список версий — в `requirements.txt`.

## Замечания по разработке

- Парсинг ответов моделей инкапсулирован в методах соответствующих классов VLM (`_parse_*`).
- Сохранение JSON на диск включается флагом **`persist=True`** у методов консолидации и VLM-этапов в `main.py`.
- Отдельные модули оценки по GT, построение кропов из масок и служебный разбор старых текстовых форматов из репозитория удалены; актуальная логика ограничена цепочкой `main.py` и перечисленными пакетами.
