from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DETAIL_MODEL_NAME
from vlm.base import VLMClient


def _compact_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()

    def s(v: Any) -> Optional[str]:
        if v is None:
            return None
        t = str(v).strip()
        return t or None

    description = s(item.get("description"))
    if description:
        return description

    label = s(item.get("label"))
    parts: List[str] = []
    for k in ("color", "material", "shape", "text_markings", "relation", "confidence"):
        v = s(item.get(k))
        if v is None:
            continue
        parts.append(f"{k}={v}")

    if label and parts:
        return f"{label} ({', '.join(parts)})"
    if label:
        return label
    if parts:
        return ", ".join(parts)
    return ""


def _parse_best_index(text: str, k: int) -> Optional[int]:
    """
    Ожидаем JSON вида {"best_mask": <int>} где int в [1..k].
    Возвращает 0-based index.
    """
    if not text:
        return None
    t = text.strip()
    try:
        obj = json.loads(t)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    v = obj.get("best_mask")
    if not isinstance(v, int):
        return None
    if not (1 <= v <= k):
        return None
    return v - 1


class SAM3MaskChooserVLM(VLMClient):
    """
    Небольшой MLLM-хелпер: выбирает лучшую маску среди K кандидатных,
    используя raw кадр и изображения-оверлеи с масками.
    """

    def __init__(self, model_name: str = DETAIL_MODEL_NAME):
        super().__init__(model_name=model_name)

    def query(self, image_paths: List[Path], item: Any) -> str:
        item_text = _compact_item_text(item)
        k = max(0, len(image_paths) - 1)  # raw + K masks
        prompt_text = f"""You are helping to pick the correct segmentation mask.

        You are shown:
        1) The raw image
        2) Then {k} candidate images, each is the raw image with ONE candidate mask overlaid.

        Target description:
        {item_text}

        Task:
        Pick exactly one candidate mask image (from 1 to {k}) that best matches the target description.

        Output format (strict JSON):
        {{"best_mask": <int>}}
        """
        return self._run_inference(image_paths=image_paths, prompt_text=prompt_text)

    def choose_best(self, raw_image_path: Path, overlay_paths: List[Path], item: Any) -> Optional[int]:
        """
        Возвращает 0-based индекс в overlay_paths.
        """
        if not overlay_paths:
            return None
        # MAX_CROPS_PER_REQUEST = 5, поэтому raw + до 4 оверлеев
        overlay_paths = overlay_paths[:4]
        text = self.query([raw_image_path, *overlay_paths], item=item)
        return _parse_best_index(text, k=len(overlay_paths))

