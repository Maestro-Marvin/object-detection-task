import numpy as np
from typing import List, Dict, Tuple

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