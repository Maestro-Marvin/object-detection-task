from pathlib import Path
from typing import List, Optional

from config import TASK_MODEL_NAME
from .base import SharedVLMEngine, VLMClient


class SupportDiscoveryVLM(VLMClient):
    """
    Ищет опорные объекты напрямую по кадрам (без GT-масок).
    """

    def __init__(self, model_name: str = TASK_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)
        self.sampling_params.max_tokens = 512

    def query(self, image_paths: List[Path]) -> str:
        prompt_text = f"""You are an expert in scene understanding.
You are shown {len(image_paths)} representative frames from one video scene.

Task:
Identify stable SUPPORT objects (surfaces/containers/furniture) that can have other objects on/inside/near them.

Examples of valid support objects:
- desk, table, countertop, shelf, cabinet, bed, sofa, chair, sink, rack, cart

Rules:
- Return 3-10 generic support object labels in lowercase singular.
- Prefer persistent, structurally stable objects that appear in multiple views.
- Do NOT return tiny movable items (book, bottle, mouse, etc.).
- Use generic names, not brands.

Output format (strict JSON only):
["desk", "shelf", "countertop"]
"""
        return self._run_inference(image_paths, prompt_text)

