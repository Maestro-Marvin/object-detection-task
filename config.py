from pathlib import Path
import json

DATA_ROOT = Path("scenes/scene2")
FRAMES_DIR = DATA_ROOT / "rgb"
MASKS_DIR = DATA_ROOT / "gt_instance_iphone/render_instance_npy"
DESC_PATH = DATA_ROOT / "gt_categories.json"
CROPS_DIR = Path("crops")
GT_JSON = Path("results/ground_truth.json")
PRED_JSON = Path("results/predictions.json")
DETAILED_PRED_JSON = Path("results/detailed_predictions.json")
REPORT_JSON = Path("results/report.json")
METRICS_JSON = Path("results/metrics.json")
TEMP_GT_JSON = Path("results/temp_gt.json")
SELECTED_CROPS = Path("results/selected_crops.json")

SUPPORT_IDS_PATH = DATA_ROOT / "support_ids.json"

def _load_support_ids() -> set[int]:
    with open(SUPPORT_IDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(x) for x in data}

SUPPORT_OBJECT_IDS = _load_support_ids()

MIN_BBOX_RATIO = 0.05
PADDING_RATIO_GT = 0.5
PADDING_RATIO_MODEL = 0.25
MASK_COLOR = (255, 0, 255)
BACKGROUND_ID = -100
FRAMES_SHARE = 0.5
SIMILARITY_THRESHOLD = 0.65

TASK_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
SELECTOR_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DETAIL_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MAX_CROPS_PER_REQUEST = 5

# --- SAM3 localization (Ultralytics) ---
LOCALIZATION_DIR = Path("localization")
SAM3_MODEL_PATH = Path("sam3/weights/sam3.pt")
SAM3_CONF = 0.25
SAM3_HALF = True
SAM3_SAVE_BINARY_MASKS = True

# --- SAM3 Agent (MLLM chooser) ---
# Сколько кандидатных масок показывать MLLM для выбора (top-K).
SAM3_AGENT_TOPK = 3
