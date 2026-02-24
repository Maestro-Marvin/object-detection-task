import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
from config import FRAMES_SHARE, PADDING_RATIO_GT
from support_object_utils import expand_bbox


def is_bbox_inside(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
    x1_in, y1_in, x2_in, y2_in = inner
    x1_out, y1_out, x2_out, y2_out = outer
    return x1_out <= x1_in and y1_out <= y1_in and x2_in <= x2_out and y2_in <= y2_out

class GTBuilder:
    def __init__(self, descriptions: Dict[int, str], threshold: float = FRAMES_SHARE):
        self.descriptions = descriptions
        self.threshold = threshold
        self.gt_occurrences = defaultdict(lambda: defaultdict(int))  # support_id → small_id → count
        self.total_frames = defaultdict(int)

    def process_frame(self, mask: np.ndarray, supports: list[dict]):
        """Обрабатывает один кадр и обновляет внутренние счётчики."""

        # Строим bbox для всех объектов на кадре
        all_ids = np.unique(mask)
        all_ids = all_ids[all_ids != 0]
        all_bboxes = {}
        for obj_id in all_ids:
            ys, xs = np.where(mask == obj_id)
            if xs.size > 0:
                all_bboxes[obj_id] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # Для каждого опорного объекта - ищем мелкие внутри расширенного bbox
        for obj in supports:
            support_id = obj["id"]
            support_bbox = obj["bbox"]
            support_bbox_expanded = expand_bbox(support_bbox, mask.shape, PADDING_RATIO_GT)

            self.total_frames[support_id] += 1

            for small_id, small_bbox in all_bboxes.items():
                if small_id == support_id:
                    continue
                if is_bbox_inside(small_bbox, support_bbox_expanded):
                    self.gt_occurrences[support_id][small_id] += 1

    def build_final_gt(self) -> Dict[int, list[str]]:
        """Формирует финальный GT после обработки всех кадров."""
        final_gt = {}
        for support_id, small_counts in self.gt_occurrences.items():
            total = self.total_frames[support_id]
            min_count = int(self.threshold * total)
            stable_small_ids = [
                sid for sid, cnt in small_counts.items() if cnt >= min_count
            ]
            if stable_small_ids:
                final_gt[support_id] = [
                    self.descriptions[sid] for sid in stable_small_ids if self.descriptions[sid] != f"unknown_{sid}"
                ]
        return final_gt