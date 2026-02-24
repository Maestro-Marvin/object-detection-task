from pathlib import Path

DATA_ROOT = Path("data")
FRAMES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "render_instance_npy"
DESC_PATH = DATA_ROOT / "gt_categories.json"
CROPS_DIR = Path("crops")
GT_JSON = Path("results/ground_truth.json")
PRED_JSON = Path("results/predictions.json")
REPORT_JSON = Path("results/report.json")
METRICS_JSON = Path("results/metrics.json")

SUPPORT_KEYWORDS = {
    "table", "bed", "shelf", "bookshelf", "shelves",
    "floor", "ceiling", "countertop", "cabinet", "cabinets",
    "chair", "drawers", "carpet", "container", "box"
}

MIN_BBOX_RATIO = 0.05
PADDING_RATIO_GT = 0.5
PADDING_RATIO_MODEL = 0.25
MASK_COLOR = (128, 128, 128)
FRAMES_SHARE = 0.5

VLM_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-FP8"
LLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
MAX_CROPS_PER_REQUEST = 5