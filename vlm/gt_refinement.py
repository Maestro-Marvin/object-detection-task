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

        Below is a list of candidate items that might be associated with this object:
        Candidates: [{candidates_str}]

        Your task: determine which of these candidates are **truly present and correctly associated** with the target object.

        For each valid item, assign the correct preposition:
        - 'on' if it's placed ON TOP of a surface (e.g., lamp on desk),
        - 'inside' if it's physically INSIDE a container (e.g., book inside cabinet),
        - 'near' if it's NEXT TO or BESIDE the object (e.g., slippers near bed).

        Note: Gray regions (#808080) belong to other support objects — ignore them completely.

        Rules:
        - ONLY include items from the candidate list.
        - NEVER include the main object itself ('{description}').
        - Group by preposition.
        - Output format: '[item1, item2] on {description} id={obj_id}; [item3] inside {description} id={obj_id}; ...'
        - If NO candidates are valid, output exactly: 'none'
        - Use lowercase English words, no quotes, no extra punctuation.
        - Return ONLY the string. No explanations, no JSON, no markdown.

        Examples of GOOD output:
        [lamp, notebook] on desk id=42
        [shampoo] inside cabinet id=70; [slippers] near cabinet id=70
        none

        Now analyze the images and return your answer."""


        return self._run_inference(image_paths, prompt_text)