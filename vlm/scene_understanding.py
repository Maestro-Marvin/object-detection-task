from .base import VLMClient
from PIL import Image
from pathlib import Path
from typing import List

class SceneUnderstandingVLM(VLMClient):
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