from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DETAIL_MODEL_NAME
from vlm.base import VLMClient
from vlm.base import SharedVLMEngine

import re
from typing import Optional

def _parse_best_index(text: str, k: int) -> Optional[int]:
    """
    Извлекает число из ответа модели.
    """
    if not text:
        return None
    
    # 1. Попытка найти первое целое число в тексте (Regex)
    # Ищем число, которое находится в диапазоне [1, k]
    matches = re.findall(r'\b(\d+)\b', text)
    
    for match in matches:
        val = int(match)
        if 1 <= val <= k:
            return val - 1  # Конвертируем в 0-based index
    
    # 2. Если числа не найдены или они вне диапазона
    return None


class SAM3MaskChooserVLM(VLMClient):
    """
    Небольшой MLLM-хелпер: выбирает лучшую маску среди K кандидатных,
    используя raw кадр и изображения-оверлеи с масками.
    """

    def __init__(self, model_name: str = DETAIL_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name=model_name, shared=shared)

    def query(self, image_paths: List[Path], item: Any) -> str:
        item_text = item["description"]
        k = len(image_paths) - 1  # raw + K masks
        prompt_text = f"""You are helping to pick the correct segmentation mask.

        You are shown:
        1) The raw image
        2) Then {k} candidate images, each is the raw image with ONE candidate mask overlaid.

        Target description:
        {item_text}

        Task:
        Pick exactly one candidate mask image (from 1 to {k}) that best matches the target description.

        Output format:
        Return ONLY a single integer number between 1 and {k}.
        No JSON, no brackets, no explanations, no text.

        Examples:
        1
        2
        3

        Now analyze the images and return the number:"""
        return self._run_inference(image_paths=image_paths, prompt_text=prompt_text)

    def choose_best(self, raw_image_path: Path, overlay_paths: List[Path], item: Any) -> Optional[int]:
        """
        Возвращает 0-based индекс в overlay_paths.
        """
        if not overlay_paths:
            return None
        text = self.query([raw_image_path, *overlay_paths], item=item)
        return _parse_best_index(text, k=len(overlay_paths))

