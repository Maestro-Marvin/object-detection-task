import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict

def load_descriptions(json_path: Path) -> Dict[int, str]:
    """
    Загружает описания объектов из JSON-файла в формате:
        { 
            object_id: primary_synonym
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = {}
    for sample in data.get("dataset", {}).get("samples", []):
        obj_id = sample["object_id"]
        synonyms = sample["labels"]["image_attributes"].get("synonyms", [])
        desc[obj_id] = synonyms[0].strip() if synonyms else f"unknown_{obj_id}"
    return desc

def load_frame_and_mask(frame_name: str, frames_dir: Path, masks_dir: Path):
    rgb = np.array(Image.open(frames_dir / frame_name).convert("RGB"))
    mask_path = masks_dir / (frame_name.split(".")[0] + ".npy")
    try:
        mask = np.load(mask_path)
    except FileNotFoundError:
        mask = None
    return rgb, mask

