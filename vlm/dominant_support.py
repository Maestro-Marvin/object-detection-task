from pathlib import Path
from typing import List, Optional

from config import TASK_MODEL_NAME
from .base import SharedVLMEngine, VLMClient


class DominantSupportVLM(VLMClient):
    """
    Определяет доминирующий опорный объект на кадре.
    Возвращает строго JSON: строка (label) или null.
    """

    def __init__(self, model_name: str = TASK_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)
        self.sampling_params.max_tokens = 128

    def query(self, image_paths: List[Path]) -> str:
        prompt_text = """You are given ONE image frame from a household scene.

Task:
Identify the SINGLE dominant support object in the frame: the main surface/container/furniture that is most central and useful for describing items on/inside/near it.

Valid examples:
desk, table, countertop, shelf, cabinet, bed, sofa, chair, sink

Rules:
- Return ONE generic label in lowercase singular, or null if no clear support dominates.
- Do NOT return small movable items (book, bottle, keyboard, etc.).
- Prefer the support that is most central and most likely to have associated items.

Output format (STRICT JSON only):
"desk"
or
null
"""
        return self._run_inference(image_paths, prompt_text)

