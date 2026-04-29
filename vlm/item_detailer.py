import json
from pathlib import Path
from typing import List, Optional

from config import DETAIL_MODEL_NAME
from .base import VLMClient
from .base import SharedVLMEngine


class ItemDetailerVLM(VLMClient):
    """
    VLM, которая по кропам опорного объекта и списку предсказанных предметов
    возвращает детальные описания (форма/цвет/текстура/надписи).
    """

    def __init__(self, model_name: str = DETAIL_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)
        self.sampling_params.max_tokens = 2048

    def query(self, image_paths: List[Path], support_description: str, predicted_labels: List[str]) -> str:
        if not predicted_labels:
            return "No associated objects were predicted."

        predicted_labels_str = ", ".join(predicted_labels)

        prompt_text = f"""You are an expert visual describer for scene understanding. You are shown {len(image_paths)} recent views of the same SUPPORT object: '{support_description}'.

        ### YOUR TASK:
        For each item label in the input list, generate a structured visual description based on what is CLEARLY visible in the images.

        ### INPUT LABELS:
        Below is a list of item labels that were previously identified as associated with this support object:
        Labels: [{predicted_labels_str}]

        ### DESCRIPTION GUIDELINES:
        For each label, describe visual properties ONLY when clearly visible:
        - **label**: Use the exact label from the input list (do not change it)
        - **relation**: Spatial relationship to the support object — choose ONE: "on", "inside", or "near"
        - **shape**: Geometry/form (e.g., cylindrical, rectangular, flat, curved, spherical) — use null if not visible
        - **material**: Texture/appearance (e.g., glossy plastic, matte fabric, brushed metal, paper, ceramic) — use null if not visible
        - **color**: Dominant color(s) using common color words (e.g., white, black, red, blue, transparent) — use null if not visible
        - **text_markings**: Any readable text, logos, brand names, or distinctive symbols — use null if none visible or unreadable
        - **description**: A single short natural-language description string (not a list) capturing what is clearly visible.
          If the item is not clearly visible, set description to "not clearly visible".
        - **confidence**: Assessment of visibility — "high" (clearly visible), "medium" (partially visible), or "low" (uncertain)

        ### TEXT DESCRIPTION FORMAT (for "description"):
        For each item label in the input list, generate a detailed visual description based on what is CLEARLY visible in the images.
        Examples:
          Input: ["bottle", "notebook", "cup"]
          Output: ["white cylindrical bottle with matte plastic surface and pump dispenser", "black rectangular spiral-bound notebook with plain cover", "not clearly visible"]
          Input: ["lamp", "keys"]
          Output: ["metal desk lamp with adjustable arm, dark gray finish", "metallic keychain with two keys, slightly blurred"]

        ### IMPORTANT:
        - If a property is not visible, use null — do NOT invent details
        - If an item from the list is NOT clearly visible, set confidence to "low" and use null for other fields
        - Maintain the same order as the input labels
        - The input list is a reference — describe what you actually see

        ### MASKING INSTRUCTION:
        IMPORTANT: Bright magenta areas (#FF00FF) are MASKED regions belonging to OTHER objects.
        - COMPLETELY IGNORE these areas.
        - Do NOT describe anything within magenta regions.
        - Focus only on the central support object and its associated items.

        ### OUTPUT (STRICT JSON):
        - Return a JSON array of objects.
        - Each object must have exactly these keys: "label", "relation", "shape", "material", "color", "text_markings", "description", "confidence"
        - Use null (not "null" string) for missing properties.
        - Return ONLY valid JSON. No explanations, no markdown, no code blocks.

        ### EXAMPLES:

        Input: ["bottle", "notebook", "cup"]
        Output:
        [
        {{
            "label": "bottle",
            "relation": "on",
            "shape": "cylindrical with pump dispenser",
            "material": "matte plastic",
            "color": "white with blue accents",
            "text_markings": null,
            "description": "white cylindrical bottle with matte plastic surface and pump dispenser",
            "confidence": "high"
        }},
        {{
            "label": "notebook",
            "relation": "on",
            "shape": "rectangular, spiral-bound",
            "material": "paper cover",
            "color": "black",
            "text_markings": null,
            "description": "black rectangular spiral-bound notebook with plain cover",
            "confidence": "high"
        }},
        {{
            "label": "cup",
            "relation": "near",
            "shape": null,
            "material": null,
            "color": null,
            "text_markings": null,
            "description": "not clearly visible",
            "confidence": "low"
        }}
        ]

        Input: ["lamp", "speaker"]
        Output:
        [
        {{
            "label": "lamp",
            "relation": "on",
            "shape": "adjustable arm desk lamp",
            "material": "metal",
            "color": "dark gray",
            "text_markings": null,
            "description": "metal desk lamp with adjustable arm, dark gray finish",
            "confidence": "high"
        }},
        {{
            "label": "speaker",
            "relation": "on",
            "shape": "cylindrical",
            "material": "fabric mesh",
            "color": "black",
            "text_markings": "amazon alexa",
            "description": "black cylindrical speaker covered in fabric mesh with visible 'amazon alexa' marking",
            "confidence": "high"
        }}
        ]

        Input: ["keys"]
        Output:
        [
        {{
            "label": "keys",
            "relation": "near",
            "shape": null,
            "material": "metallic",
            "color": "silver",
            "text_markings": null,
            "description": "metallic keychain with keys, slightly blurred",
            "confidence": "medium"
        }}
        ]

        Now analyze the images and return your answer."""

        return self._run_inference(image_paths, prompt_text)

