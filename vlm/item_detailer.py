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

    def query(self, image_paths: List[Path], support_description: str, predicted_labels: List[str]) -> str:
        if not predicted_labels:
            return "No associated objects were predicted."

        predicted_labels_str = ", ".join(predicted_labels)

        prompt_text = f"""You are an expert visual describer for scene understanding. You are shown {len(image_paths)} recent views of the same SUPPORT object: '{support_description}'.

        ### YOUR TASK:
        For each item label in the input list, generate a detailed visual description based on what is CLEARLY visible in the images.

        ### INPUT LABELS:
        Below is a list of item labels that were previously identified as associated with this support object:
        Labels: [{predicted_labels_str}]

        ### DESCRIPTION GUIDELINES:
        For each label, describe visual properties ONLY when clearly visible:
        - shape or form (e.g., cylindrical, rectangular, flat, curved)
        - material or texture (e.g., glossy plastic, matte fabric, brushed metal)
        - dominant colors
        - any readable text, logos, or distinctive markings

        ### IMPORTANT:
        - If a property is not visible, omit it — do NOT invent details
        - If an item from the list is NOT clearly visible, write: "not clearly visible"
        - Keep each description concise (1-2 sentences max)
        - The input list is a reference — describe what you actually see

        ### MASKING INSTRUCTION:
        IMPORTANT: Bright magenta areas (#FF00FF) are MASKED regions belonging to OTHER objects.
        - COMPLETELY IGNORE these areas.
        - Do NOT describe anything within magenta regions.
        - Focus only on the central support object and its associated items.

        ### OUTPUT (STRICT JSON):
        - Return a JSON array of strings, one description per input label.
        - Maintain the same order as the input labels.
        - Format: ["description1", "description2", ...]
        - If a label cannot be matched to any visible item, use: "not clearly visible"
        - Return ONLY valid JSON. No explanations, no markdown.

        ### EXAMPLES:
        Input: ["bottle", "notebook", "cup"]
        Output: ["white cylindrical bottle with matte plastic surface and pump dispenser", "black rectangular spiral-bound notebook with plain cover", "not clearly visible"]

        Input: ["lamp", "keys"]
        Output: ["metal desk lamp with adjustable arm, dark gray finish", "metallic keychain with two keys, slightly blurred"]

        Now analyze the images and return your answer."""

        return self._run_inference(image_paths, prompt_text)

