import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from config import TASK_MODEL_NAME

from .base import SharedVLMEngine, VLMClient


class SupportGrouperVLM(VLMClient):
    """
    Для одного кадра определяет:
    - есть ли доминирующий опорный объект (present)
    - если да, то какой (label)

    Выход: строгий JSON объект:
      {"present": true, "label": "desk"}
      {"present": false, "label": null}
    """

    def __init__(self, model_name: str = TASK_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)
        self.sampling_params.max_tokens = 128

    def query(self, image_paths: List[Path]) -> str:
        prompt_text = """You are given ONE image frame from a household scene.

Task:
Decide whether there is a SINGLE dominant support object in the frame (a main surface/container/furniture),
and if yes, name it.

Support object examples:
desk, table, countertop, shelf, cabinet, bed, sofa, chair, sink, drawer, nightstand

Rules:
- If a clear single dominant support exists: present=true and label=<one generic lowercase singular support label>.
- If NO clear dominant support exists: present=false and label=null.
- Never output small movable items (book, bottle, keyboard, etc.).
- Use generic names, not brands.

Output format (STRICT JSON only):
{"present": true, "label": "desk"}
or
{"present": false, "label": null}
"""
        return self._run_inference(image_paths, prompt_text)

    @staticmethod
    def _parse_support_reply(raw: Any) -> Tuple[bool, Optional[str]]:
        """
        Парсит JSON вида {"present": true/false, "label": "desk"|null}.
        Возвращает (present, label).
        """
        if raw is None:
            return (False, None)

        text = str(raw).strip()
        if not text:
            return (False, None)

        try:
            parsed = json.loads(text)
        except Exception:
            return (False, None)

        if not isinstance(parsed, dict):
            return (False, None)

        present = bool(parsed.get("present", False))
        label = parsed.get("label", None)
        if not present:
            return (False, None)
        if label is None:
            return (True, None)
        if isinstance(label, str):
            s = label.strip().lower()
            return (True, s or None)
        return (True, None)

    def classify_frame(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """Ответ модели сразу парсится в (present, label)."""
        try:
            raw = self.query([image_path])
            return self._parse_support_reply(raw)
        except Exception:
            return (False, None)

