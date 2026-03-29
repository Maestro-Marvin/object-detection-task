from .base import VLMClient
from PIL import Image
from pathlib import Path
from typing import List

class GTRefinementVLM(VLMClient):
    def query(self, image_paths: List[Path], support_description: str, candidates: List[str]) -> str:
        if not candidates:
            return "[]"

        candidates_str = ", ".join(candidates)
        prompt_text = f"""You are an expert in spatial scene understanding. You are shown {len(image_paths)} recent views of the same SUPPORT object: '{support_description}'.

        ### YOUR TASK:
        Identify ALL clearly visible items that have a DIRECT spatial relationship with this support object (on it, inside it, or near it).
        
        ### REFERENCE CANDIDATES (AUXILIARY):
        Below is a list of items that MIGHT be associated with this object. Use it as a helpful hint, NOT as a strict constraint:
        Candidates: [{candidates_str}]

        ### IMPORTANT:
        - If you see an item NOT in the list but clearly visible → INCLUDE it.
        - If an item IS in the list but NOT visible or wrongly associated → EXCLUDE it.
        - The candidate list is a suggestion, not a ground truth.

        ### MASKING INSTRUCTION:
        IMPORTANT: Bright magenta areas (#FF00FF) are MASKED regions belonging to OTHER objects.
        - COMPLETELY IGNORE these areas.
        - Do NOT describe or identify anything within magenta regions.
        - Focus only on the central support object and its associated items.

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