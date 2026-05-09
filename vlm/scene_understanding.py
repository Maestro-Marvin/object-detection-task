import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PRED_JSON
from utils.aggregator import save_result

from .base import SharedVLMEngine, VLMClient

class SceneUnderstandingVLM(VLMClient):
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)

    def query(self, image_paths: List[Path], support_description: str) -> str:
        prompt_text = f"""You are an expert in spatial scene understanding. You are shown {len(image_paths)} recent views of the same SUPPORT object: '{support_description}'.

        Identify ALL clearly visible items that have a DIRECT spatial relationship with this support object (on it, inside it, or near it).

        ### IMPORTANT: Bright magenta areas (#FF00FF) are MASKED regions belonging to OTHER objects.
        - COMPLETELY IGNORE these areas.
        - Do NOT describe or identify anything within magenta regions.
        - Focus only on the central support object and its associated items.

        ### NAMING CONVENTION (CRITICAL):
        - Use GENERIC CATEGORY names, NOT brand names or specific products.
        - BAD: "nivea men", "iphone", "kleenex"
        - GOOD: "shampoo bottle", "smartphone", "tissue box"
        - Always use singular form (e.g., "book" not "books").
        - Use lowercase English words.

        ### OUTPUT (STRICT JSON):
        - NEVER include the support object itself '{support_description}'.
        - Return a JSON array of strings, e.g. ["lamp", "notebook"].
        - If NO associated items are visible, return an empty JSON array: [].
        - Return ONLY valid JSON. No explanations, no markdown.

        ### EXAMPLES:
        ["lamp", "notebook"]
        ["shampoo bottle", "slippers"]
        []

        Now analyze the images and return your answer."""

        return self._run_inference(image_paths, prompt_text)

    @staticmethod
    def _parse_associated_items_list(raw: Any) -> List[str]:
        if isinstance(raw, list):
            return [str(x).strip().lower() for x in raw if str(x).strip()]
        if raw is None:
            return []
        text = str(raw).strip()
        if not text or text.lower() in ("none", "[]"):
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(x).strip().lower() for x in parsed if isinstance(x, str) and x.strip()]

    def predict_associated_items(
        self,
        selected_by_support: Dict[str, List[Path]],
        *,
        persist: bool = False,
        output_path: Optional[Path] = None,
    ) -> Dict[str, List[str]]:
        """Ключи — названия опорных объектов; при persist сохраняет JSON с теми же ключами."""
        log = logging.getLogger(__name__)
        out: Dict[str, List[str]] = {}
        for support_label, selected in selected_by_support.items():
            log.info(f"Querying task VLM for support '{support_label}' ({len(selected)} crops)")
            try:
                raw = self.query(selected, support_label)
                out[support_label] = self._parse_associated_items_list(raw)
            except Exception:
                out[support_label] = []
        if persist:
            save_result(out, output_path if output_path is not None else PRED_JSON)
        return out