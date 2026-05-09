from pathlib import Path
from typing import List, Optional

from config import SELECTOR_MODEL_NAME
from .base import SharedVLMEngine, VLMClient


class FrameSelectorVLM(VLMClient):
    """
    Турнирный селектор "лучших кадров сцены".

    Критерий: кадр лучше, если на нём хорошо виден ОДИН опорный объект
    (поверхность/контейнер/мебель) и связанные с ним предметы вокруг/на/внутри.
    """

    def __init__(self, model_name: str = SELECTOR_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)

    def query(self, image_paths: List[Path], description: str, obj_id: int) -> str:
        support_hint = str(description).strip().lower()
        support_line = f"Target support object: '{support_hint}'.\n" if support_hint else ""

        prompt_text = f"""You are selecting the best frame for a downstream pipeline.
You are shown two frames (A then B) from the same scene.

{support_line}
Goal:
Pick the frame that is BEST for understanding the target support object (if specified) and the items associated with it.

What "best" means (priority order):
1) The target support object is clearly visible (or, if not specified, ONE dominant support is clearly visible).
2) Items on/inside/near that support object are also visible and not heavily occluded.
3) Frame is sharp (low motion blur), well-focused, good exposure.
4) Camera is not too close / not too far; enough context around the support object.

Avoid frames where:
- multiple competing supports dominate equally (confusing),
- the main support is heavily occluded,
- strong motion blur / darkness.

Output format:
Return exactly one character: A or B. No other text."""

        out = self._run_inference(image_paths, prompt_text).strip().upper()
        if "A" in out and "B" not in out:
            return "A"
        if "B" in out and "A" not in out:
            return "B"
        if out.startswith("B"):
            return "B"
        return "A"

