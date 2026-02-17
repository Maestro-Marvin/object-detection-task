import numpy as np
from typing import List, Dict
from config import SUPPORT_KEYWORDS, MIN_BBOX_RATIO

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