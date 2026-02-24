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

        Note: Some regions in the image are masked in gray (#808080). These areas belong to other support objects and should be ignored. Focus only on visible items related to the target object - '{description}'.

        Rules:
        - NEVER include the main object itself '{description}'.
        - Group items by preposition.
        - Output format: '[item1, item2] on {description} id={obj_id}; [item3] inside {description} id={obj_id}; ...'
        - If multiple prepositions apply, list all of them, separated by '; '.
        - If NO related items are visible, output exactly: 'none'
        - Use lowercase English words, no quotes, no extra punctuation.
        - Return ONLY the string. No explanations, no JSON, no markdown.

        Examples of GOOD output:
        [lamp, notebook] on desk id=42
        [shampoo] inside cabinet id=70; [slippers] near cabinet id=70
        none

        Now analyze the images and return your answer."""

        return self._run_inference(image_paths, prompt_text)