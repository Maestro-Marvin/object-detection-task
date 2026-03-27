import json
from pathlib import Path
from typing import List

from config import DETAIL_MODEL_NAME
from .base import VLMClient


class ItemDetailerVLM(VLMClient):
    """
    VLM, которая по кропам опорного объекта и списку предсказанных предметов
    возвращает детальные описания (форма/цвет/текстура/надписи).
    """

    def __init__(self, model_name: str = DETAIL_MODEL_NAME):
        super().__init__(model_name)
        self.sampling_params.max_tokens = 2048

    def query(self, image_paths: List[Path], support_description: str, obj_id: int, predicted_labels: List[str]) -> str:
        labels = [str(x).strip().lower() for x in (predicted_labels or []) if str(x).strip()]
        if not labels:
            return "No associated objects were predicted."

        labels_json = json.dumps(labels, ensure_ascii=False)

        prompt_text = f"""You are a meticulous visual describer for scene understanding. You are shown {len(image_paths)} recent views of the same support object: '{support_description}' (ID: {obj_id}).

        You are given a list of previously predicted item labels associated with this support object.

        ### IMPORTANT MASKING NOTE:
        Bright magenta regions (#FF00FF) correspond to OTHER objects and are masked.
        - COMPLETELY IGNORE magenta regions.
        - Do NOT describe anything inside them.

        ### INPUT LABELS (from a previous model):
        {labels_json}

        ### YOUR TASK:
        For each item label, write a short, natural description of what you see. Focus on visual properties that are CLEARLY visible in the images.

        Describe when visible:
        - shape or form (e.g., cylindrical, rectangular, flat, curved)
        - material or texture (e.g., glossy plastic, matte fabric, brushed metal)
        - dominant colors
        - any readable text, logos, or distinctive markings

        ### OUTPUT FORMAT:
        Return plain text in English. Use one concise sentence or short paragraph per item.
        Format each line as:
        • [label]: [description]

        Keep descriptions factual and brief (1-2 sentences max per item).
        Do NOT invent details — if a property is not visible, simply omit it.
        Do NOT use brand names — use generic terms (e.g., "bottle" not "nivea").

        If an item is not clearly visible or seems incorrectly labeled, write:
        • [label]: not clearly visible

        ### EXAMPLES OF GOOD OUTPUT:
        • bottle: white cylindrical bottle with matte plastic surface and pump dispenser, no visible text
        • notebook: black rectangular spiral-bound notebook with plain cover
        • cup: small ceramic cup, light blue color, partially obscured
        • shampoo: not clearly visible

        ### RULES:
        - Return ONLY the descriptions, no introductory text or explanations
        - Use lowercase for labels, normal capitalization for descriptions
        - Separate items with line breaks
        - If the list is empty or no items are visible, output: "no items clearly visible"

        Now analyze the images and return your answer."""

        return self._run_inference(image_paths, prompt_text)

