import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from config import PADDING_RATIO

def save_crop(
    rgb_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    obj_id: int,
    frame_id: str,
    crops_dir: Path,
) -> Path:
    H, W = rgb_img.shape[:2]
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    pad_x = int(w * PADDING_RATIO)
    pad_y = int(h * PADDING_RATIO)

    new_x1 = max(0, x1 - pad_x)
    new_y1 = max(0, y1 - pad_y)
    new_x2 = min(W, x2 + pad_x)
    new_y2 = min(H, y2 + pad_y)

    crop = rgb_img[new_y1:new_y2, new_x1:new_x2]

    obj_dir = crops_dir / str(obj_id)
    obj_dir.mkdir(parents=True, exist_ok=True)
    crop_path = obj_dir / f"{frame_id}.jpg"
    cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    return crop_path