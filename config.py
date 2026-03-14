from pathlib import Path

DATA_ROOT = Path("scenes")
FRAMES_DIR = DATA_ROOT / "rgb"
MASKS_DIR = DATA_ROOT / "gt_instance_iphone/render_instance_npy"
DESC_PATH = DATA_ROOT / "gt_categories.json"
CROPS_DIR = Path("crops")
GT_JSON = Path("results/ground_truth.json")
PRED_JSON = Path("results/predictions.json")
REPORT_JSON = Path("results/report.json")
METRICS_JSON = Path("results/metrics.json")
TEMP_GT_JSON = Path("results/temp_gt.json")
SELECTED_CROPS = Path("results/selected_crops.json")

SUPPORT_KEYWORDS = {
    "table", "bed", "shelf", "bookshelf", "shelves",
    "floor", "countertop", "cabinets",
    "drawers"
}

MIN_BBOX_RATIO = 0.05
PADDING_RATIO_GT = 0.25
PADDING_RATIO_MODEL = 0.1
MASK_COLOR = (255, 0, 255)
BACKGROUND_ID = -100
FRAMES_SHARE = 0.5
SIMILARITY_THRESHOLD = 0.65

TASK_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
SELECTOR_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MAX_CROPS_PER_REQUEST = 5
