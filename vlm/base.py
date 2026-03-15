from abc import ABC, abstractmethod
import re
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple
from config import TASK_MODEL_NAME, MAX_CROPS_PER_REQUEST

REASONING_SYSTEM_SUFFIX = (
    " Answer the question in the following format: "
    "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
)


class VLMClient(ABC):
    def __init__(self, model_name: str = TASK_MODEL_NAME):
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": MAX_CROPS_PER_REQUEST},
            gpu_memory_utilization=0.3,
            max_model_len=32768,
        )
        self.sampling_params = SamplingParams(max_tokens=4096, temperature=0.0)

    def _prepare_messages(
        self,
        pil_images: List[Image.Image],
        prompt_text: str,
        system_prompt: Optional[str] = None,
    ):
        """Создаёт messages в формате Qwen3-VL / Cosmos, с опциональным system."""
        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": content})
        return messages

    def _parse_think_and_answer(self, text: str) -> Tuple[str, str]:
        """Возвращает (answer, explanation) из <think>/<answer> блоков."""
        explanation = ""
        answer = text.strip()

        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            explanation = think_match.group(1).strip()

        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        elif "</think>" in text:
            # Fallback: всё после блока рассуждений считаем ответом
            _, tail = text.split("</think>", 1)
            answer = tail.strip()

        return answer, explanation

    def _run_inference(
        self,
        image_paths: List[Path],
        prompt_text: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Общая логика генерации. Снаружи возвращаем только конечный ответ (без рассуждений)."""
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        messages = self._prepare_messages(pil_images, prompt_text, system_prompt)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, _, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
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
        raw = outputs[0].outputs[0].text.strip()
        answer, _ = self._parse_think_and_answer(raw)
        return answer

    @abstractmethod
    def query(self, *args, **kwargs) -> str:
        """Абстрактный метод — должен быть реализован в подклассах."""
        pass