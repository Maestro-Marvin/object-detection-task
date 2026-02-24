import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from support_object_utils import expand_bbox
from config import PADDING_RATIO_MODEL, MASK_COLOR

def save_crop(
    rgb_img: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    obj_id: int,
    other_support_ids: List[int],
    frame_id: str,
    crops_dir: Path,
    mask_color: Tuple[int, int, int] = MASK_COLOR
) -> Path:

    new_x1, new_y1, new_x2, new_y2 = expand_bbox(bbox, mask.shape, PADDING_RATIO_MODEL)

    crop_rgb = rgb_img[new_y1:new_y2, new_x1:new_x2].copy()
    crop_mask = mask[new_y1:new_y2, new_x1:new_x2]

    for other_id in other_support_ids:
        if other_id == obj_id:
            continue
        mask_other = (crop_mask == other_id)
        if np.any(mask_other):
            crop_rgb[mask_other] = mask_color

    obj_dir = crops_dir / str(obj_id)
    obj_dir.mkdir(parents=True, exist_ok=True)
    crop_path = obj_dir / f"{frame_id}.jpg"
    cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

    return crop_path