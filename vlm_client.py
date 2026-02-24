from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
from typing import List
from config import MODEL_NAME, MAX_CROPS_PER_REQUEST

class VLMClient:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

        self.llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": MAX_CROPS_PER_REQUEST},
            gpu_memory_utilization = 0.3,
            max_model_len = 65536,
        )
        self.sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)

    def _prepare_messages(self, pil_images: List[Image.Image], prompt_text: str):
        """Создаёт messages в формате Qwen-VL."""
        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def query(self, image_paths: List[Path], description: str, obj_id: int) -> str:
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]

        prompt_text = (
            f"You are an expert in spatial scene understanding. You are shown {len(pil_images)} recent views of the same object: '{description}' (ID: {obj_id}).\n\n"
            "Identify ALL clearly visible items that have a DIRECT spatial relationship with this object.\n\n"
            "For each item, determine the correct preposition:\n"
            "- Use 'on' if it's placed ON TOP of a surface (e.g., lamp on desk).\n"
            "- Use 'inside' if it's physically INSIDE a container (e.g., book inside shelves).\n"
            "- Use 'near' if it's NEXT TO or BESIDE the object (e.g., shoes near carpet).\n\n"
            "Note: Some regions in the image are masked in gray (#808080). These areas belong to other support objects and should be ignored. Focus only on visible items related to the target object - '{description}'.\n"
            "Rules:\n"
            "- NEVER include the main object itself '{description}'.\n"
            "- Group items by preposition.\n"
            "- Output format: '[item1, item2] on {description} id={obj_id}; [item3] inside {description} id={obj_id}; ...'\n"
            "- If multiple prepositions apply, list all of them, separated by '; '.\n"
            "- If NO related items are visible, output exactly: 'none'\n"
            "- Use lowercase English words, no quotes, no extra punctuation.\n"
            "- Return ONLY the string. No explanations, no JSON, no markdown.\n\n"
            "Examples of GOOD output:\n"
            "[lamp, notebook] on desk id=42\n"
            "[shampoo] inside cabinet id=70; [slippers] near cabinet id=70\n"
            "none\n\n"
            "Now analyze the images and return your answer."
        )

        messages = self._prepare_messages(pil_images, prompt_text)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, _, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_input = {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = self.llm.generate([llm_input], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip()

