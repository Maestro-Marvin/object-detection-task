from pathlib import Path

DATA_ROOT = Path("scenes/scene2")
FRAMES_DIR = DATA_ROOT / "rgb"
PRED_JSON = Path("results/predictions.json")
DETAILED_PRED_JSON = Path("results/detailed_predictions.json")
FRAMES_BY_SUPPORT_RAW_JSON = Path("results/frames_by_support_raw.json")
FRAMES_BY_SUPPORT_JSON = Path("results/frames_by_support.json")
SELECTED_CROPS = Path("results/selected_crops.json")

TASK_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
SELECTOR_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DETAIL_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_CROPS_PER_REQUEST = 5

# --- SAM3 localization (Ultralytics) ---
LOCALIZATION_DIR = Path("localization")
SAM3_MODEL_PATH = Path("sam3/weights/sam3.pt")
SAM3_CONF = 0.25
SAM3_HALF = True
SAM3_SAVE_BINARY_MASKS = True

# --- SAM3 Agent (MLLM chooser) ---
SAM3_AGENT_TOPK = 3

# --- End-to-end global frame selection ---
FRAME_STRIDE = 5
