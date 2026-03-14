from .base import VLMClient
from PIL import Image
from pathlib import Path
from typing import List

class SceneUnderstandingVLM(VLMClient):
    def query(self, image_paths: List[Path], description: str, obj_id: int) -> str:
        prompt_text = f"""You are an expert in spatial scene understanding. You are shown {len(image_paths)} recent views of the same object - '{description}' (ID: {obj_id}).

        Identify ALL clearly visible items that have a DIRECT spatial relationship with this object.

        For each item, determine the correct preposition:
        - Use 'on' if it's placed ON TOP of a surface (e.g., lamp on desk).
        - Use 'inside' if it's physically INSIDE a container (e.g., book inside shelves).
        - Use 'near' if it's NEXT TO or BESIDE the object (e.g., shoes near carpet).

       ### IMPORTANT: Bright magenta areas (#FF00FF) are MASKED regions belonging to OTHER objects.
        - COMPLETELY IGNORE these areas.
        - Do NOT describe or identify anything within magenta regions.
        - Focus only on the central reference object and items directly on/near it.

        ### NAMING CONVENTION (CRITICAL):
        - Use GENERIC CATEGORY names, NOT brand names or specific products.
        - BAD: "nivea men", "iphone", "kleenex"
        - GOOD: "shampoo bottle", "smartphone", "tissue box"
        - Always use singular form (e.g., "book" not "books").
        - Use lowercase English words.

        ### RULES:
        - NEVER include the main object itself '{description}'.
        - Group items by preposition.
        - Output format: '[item1, item2] on {description} id={obj_id}; [item3] inside {description} id={obj_id}; ...'
        - If multiple prepositions apply, list all of them, separated by '; '.
        - If NO related items are visible, output exactly: 'none'
        - Return ONLY the string. No explanations, no JSON, no markdown.

        ### EXAMPLES OF GOOD OUTPUT:
        [lamp, notebook] on desk id=42
        [shampoo bottle] inside cabinet id=70; [slippers] near cabinet id=70
        [smartphone, keys] on table id=15
        none

        Now analyze the images and return your answer."""

        return self._run_inference(image_paths, prompt_text)