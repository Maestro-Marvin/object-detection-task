import json
import logging
from pathlib import Path

from config import CROPS_DIR, DESC_PATH, SELECTED_CROPS
from pipeline.base import PipelineStage
from support_objects.select_best_crops import select_best_crops_tournament
from utils.aggregator import collect_crops_by_object
from utils.data_loader import load_descriptions
from vlm.base import SharedVLMEngine
from vlm.crop_selector import CropSelectorVLM


class SelectCropsStage(PipelineStage):
    def __init__(
        self,
        shared_vlm: SharedVLMEngine | None = None,
        desc_path: Path = DESC_PATH,
        crops_dir: Path = CROPS_DIR,
        output_path: Path = SELECTED_CROPS,
    ):
        self.shared_vlm = shared_vlm
        self.desc_path = desc_path
        self.crops_dir = crops_dir
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    def _load_cache(self) -> dict:
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def run(self) -> None:
        if self.shared_vlm is None:
            raise RuntimeError("shared_vlm is required to run SelectCropsStage")

        logger = logging.getLogger(__name__)
        logger.info("Selecting best crops...")

        descriptions = load_descriptions(self.desc_path)
        object_crops = collect_crops_by_object(self.crops_dir)
        selected_crops_cache = self._load_cache()
        vlm_selector = CropSelectorVLM(shared=self.shared_vlm)

        for obj_id, crop_paths in object_crops.items():
            desc = descriptions[obj_id][0]
            cache_key = str(obj_id)
            if cache_key in selected_crops_cache:
                continue

            try:
                selected_paths = select_best_crops_tournament(
                    crop_paths, vlm_selector, desc, obj_id
                )
            except Exception:
                selected_paths = []

            selected_crops_cache[cache_key] = [str(p) for p in selected_paths]
            self.save(selected_crops_cache)

        logger.info("Selected crops saved to %s", self.output_path)

    def load_output(self) -> dict[int, list[Path]]:
        raw = super().load_output()
        return {obj_id: [Path(p) for p in paths] for obj_id, paths in raw.items()}
