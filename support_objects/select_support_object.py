import numpy as np
from typing import List, Dict
from config import SUPPORT_OBJECT_IDS, MIN_BBOX_RATIO, BACKGROUND_ID

def select_support_objects(scene_mask: np.ndarray, descriptions: Dict[int, List[str]]) -> List[Dict]:
    """
    Отбирает опорные объекты по явному списку id и размеру bbox.

    Возвращает список словарей: 
        {
            "id": int,
            "description": str,
            "bbox": (x_min, y_min, x_max, y_max)
        }
    """
    unique_ids = np.unique(scene_mask)
    unique_ids = unique_ids[unique_ids != BACKGROUND_ID]
    supports = []
    for obj_id in unique_ids:
        obj_id_int = int(obj_id)
        if obj_id_int not in SUPPORT_OBJECT_IDS:
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
        if obj_id_int not in descriptions:
            continue
        supports.append({
            "id": obj_id_int,
            "description": descriptions[obj_id_int][0],
            "bbox": (int(x_min), int(y_min), int(x_max), int(y_max))
        })
    return supports