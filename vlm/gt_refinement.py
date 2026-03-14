from .base import VLMClient
from PIL import Image
from pathlib import Path
from typing import List

class GTRefinementVLM(VLMClient):
    def query(self, image_paths: List[Path], description: str, obj_id: int, candidates: List[str]) -> str:
        if not candidates:
            return "none"

        candidates_str = ", ".join(candidates)
        prompt_text = f"""You are an expert in spatial scene understanding. You are shown {len(image_paths)} recent views of the same object - '{description}' (ID: {obj_id}).

        ### YOUR TASK:
        Identify ALL clearly visible items that have a DIRECT spatial relationship with the target object.
        For each valid item, assign the correct preposition:
        - 'on' if it's placed ON TOP of a surface (e.g., lamp on desk),
        - 'inside' if it's physically INSIDE a container (e.g., book inside cabinet),
        - 'near' if it's NEXT TO or BESIDE the object (e.g., slippers near bed).
        
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
        - Focus only on the central reference object and items directly on/near it.

        ### OUTPUT RULES:
        - NEVER include the main object itself ('{description}').
        - Group items by preposition.
        - Output format: '[item1, item2] on {description} id={obj_id}; [item3] inside {description} id={obj_id}; ...'
        - If NO related items are visible, output exactly: 'none'
        - Return ONLY the string. No explanations, no JSON, no markdown.

        ### EXAMPLES OF GOOD OUTPUT:
        [lamp, notebook] on desk id=42
        [shampoo bottle] inside cabinet id=70; [slippers] near cabinet id=70
        [smartphone, keys, pen] on table id=15
        none

        Now analyze the images and return your answer."""

        return self._run_inference(image_paths, prompt_text)