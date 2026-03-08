import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List

def load_descriptions(json_path: Path) -> Dict[int, List[str]]:
    """
    Загружает описания объектов из JSON-файла в формате:
        { 
            object_id: synonims
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = {}
    for sample in data["dataset"]["samples"]:
        obj_id = sample["object_id"]
        synonyms = sample["labels"]["image_attributes"].get("synonyms", [])
        if synonyms != []:
            desc[obj_id] = synonyms
    return desc

def load_frame_and_mask(frame_name: str, frames_dir: Path, masks_dir: Path):
    rgb = np.array(Image.open(frames_dir / frame_name).convert("RGB"))
    mask_path = masks_dir / (frame_name.split(".")[0] + ".npy")
    try:
        mask = np.load(mask_path)
    except FileNotFoundError:
        mask = None
    return rgb, mask

