from .base import VLMClient
from pathlib import Path
from typing import List, Optional
from config import SELECTOR_MODEL_NAME
from .base import SharedVLMEngine

class CropSelectorVLM(VLMClient):

    def __init__(self, model_name: str = SELECTOR_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)

    def query(self, image_paths: List[Path], description: str, obj_id: int) -> str:

        prompt_text = f"""You are a visual quality judge for crop selection.
        You are shown two images (A then B) of the same reference object: '{description}' (ID: {obj_id}).

        TASK: Pick the image that is BETTER for identifying small objects on/near the reference object.

        Evaluation criteria (in priority order):
        1. CLARITY: Less motion blur, better focus, higher sharpness.
        2. VISIBILITY: The reference object and its surface are more visible (less occluded).
        3. LIGHTING: Better exposure (not too dark, not overexposed).
        4. CONTEXT: Shows more relevant surrounding area, but still focused on the object.

        IMPORTANT:
        - Bright magenta areas (#FF00FF) are masked regions of OTHER objects. IGNORE them when judging quality.
        - Focus only on the central reference object and items directly on/near it.

        TIE-BREAKER:
        - If both images are equal in quality, prefer the one with LESS occlusion of the central object.

        OUTPUT FORMAT:
        - Answer with exactly ONE character: A or B
        - No explanations, no punctuation, no extra text.

        Examples:
        [Image A: sharp, well-lit] vs [Image B: slightly blurry] → A
        [Image A: object 50% covered] vs [Image B: object fully visible] → B

        Now compare and answer:"""

        out = self._run_inference(image_paths, prompt_text).strip().upper()
        if "A" in out and "B" not in out:
            return "A"
        if "B" in out and "A" not in out:
            return "B"
        if out.startswith("B"):
            return "B"
        return "A"

