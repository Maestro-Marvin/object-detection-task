from abc import ABC, abstractmethod
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
from typing import List, Optional
from config import TASK_MODEL_NAME, MAX_CROPS_PER_REQUEST

@dataclass(frozen=True)
class SharedVLMEngine:
    """
    Общие ресурсы для VLM: один vLLM движок + один processor.
    Используем, когда в проекте одна и та же модель во всех стейджах.
    """
    model_name: str
    processor: any
    llm: any

    @staticmethod
    def build(model_name: str = TASK_MODEL_NAME, gpu_memory_utilization: float = 0.3) -> "SharedVLMEngine":
        processor = AutoProcessor.from_pretrained(model_name)
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": MAX_CROPS_PER_REQUEST},
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=65536,
        )
        return SharedVLMEngine(model_name=model_name, processor=processor, llm=llm)

    def shutdown(self) -> None:
        """
        Аккуратно останавливает engine core.
        """
        try:
            llm_engine = getattr(self.llm, "llm_engine", None)
            if llm_engine is not None:
                llm_engine.shutdown()
        except Exception:
            pass

class VLMClient(ABC):
    def __init__(self, model_name: str = TASK_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        if shared is not None:
            if shared.model_name != model_name:
                raise ValueError(
                    f"SharedVLMEngine model mismatch: shared={shared.model_name}, client={model_name}"
                )
            self.processor = shared.processor
            self.llm = shared.llm
            self._shared = shared
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                enforce_eager=True,
                limit_mm_per_prompt={"image": MAX_CROPS_PER_REQUEST},
                gpu_memory_utilization=0.3,
                max_model_len=32768,
            )
            self._shared = None
        self.sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)

    def _prepare_messages(self, pil_images: List[Image.Image], prompt_text: str):
        """Создаёт messages в формате Qwen-VL."""
        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def _run_inference(self,image_paths: List[Path], prompt_text: str) -> str:
        """Общая логика генерации."""
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
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

    @abstractmethod
    def query(self, *args, **kwargs) -> str:
        """Абстрактный метод — должен быть реализован в подклассах."""
        pass