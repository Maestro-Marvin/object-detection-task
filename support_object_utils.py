import numpy as np
from typing import List, Dict, Tuple
from config import SUPPORT_KEYWORDS, MIN_BBOX_RATIO

def expand_bbox(bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_ratio: float) -> Tuple[int, int, int, int]:
    H, W = img_shape
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(W, x2 + pad_x),
        min(H, y2 + pad_y)
    )

def select_support_objects_scene(descriptions: Dict[int, str]) -> List[int]:
    supports = []
    for object_id, primary_synonym in descriptions.items():
        if primary_synonym in SUPPORT_KEYWORDS:
            supports.append(object_id)
    return supports


def select_support_objects(scene_mask: np.ndarray, descriptions: Dict[int, str]) -> List[Dict]:
    """
    Отбирает опорные объекты на основе семантики и размера bbox

    Возвращает список словарей: 
        {
            "id": int,
            "description": str,
            "bbox": (x_min, y_min, x_max, y_max)
        }
    """
    unique_ids = np.unique(scene_mask)
    unique_ids = unique_ids[unique_ids != 0]
    supports = []
    for obj_id in unique_ids:
        desc = descriptions.get(obj_id, "").lower()
        if not any(kw == desc for kw in SUPPORT_KEYWORDS):
            continue
        ys, xs = np.where(scene_mask == obj_id)
        if xs.size == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w, h = x_max - x_min + 1, y_max - y_min + 1
        bbox_area = w * h
        image_area = scene_mask.shape[0] * scene_mask.shape[1]
        if bbox_area < MIN_BBOX_RATIO * image_area:
            continue
        supports.append({
            "id": int(obj_id),
            "description": descriptions[obj_id],
            "bbox": (int(x_min), int(y_min), int(x_max), int(y_max))
        })
    return supports