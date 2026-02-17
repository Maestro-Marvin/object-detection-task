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
            f"You are an expert scene analyzer. You see {len(pil_images)} recent views of the same {description} (object ID: {obj_id}).\n\n"
            "Your task: list ONLY small, distinct, clearly visible objects that are:\n"
            "- directly ON the object (if it has a surface like a table, shelf, bed), OR\n"
            "- immediately NEAR it (within ~30 cm, e.g., on the floor next to a chair).\n\n"
            "Rules:\n"
            "- NEVER include the {description} itself.\n"
            "- If you see NO such objects, return an empty JSON object: {{}}\n"
            "- If you see objects, return EXACTLY ONE JSON key-value pair:\n"
            "    Key format: \"[object1, object2, ...] on {description} id={obj_id}\"   (for horizontal surfaces)\n"
            "    OR        : \"[object1, object2, ...] near {description} id={obj_id}\"  (for vertical/walls)\n"
            "- Value must ALWAYS be an empty string: \"\"\n"
            "- Use lowercase, no quotes inside the list, no extra spaces.\n"
            "- NO explanations, NO markdown, NO extra text.\n"
            "- Return VALID JSON ONLY.\n\n"
            "Examples of GOOD output:\n"
            '{{"[laptop, pillow] on bed id=127": ""}}\n'
            '{{"[bottle, phone] on bookshelf id=73": ""}}\n'
            "{{}}\n\n"
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

