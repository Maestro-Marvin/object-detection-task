import json
import logging
from pathlib import Path

from config import DESC_PATH, PRED_JSON
from pipeline.base import PipelineStage
from pipeline.select_crops import SelectCropsStage
from utils.data_loader import load_descriptions
from utils.prediction_parser import safe_json_list
from vlm.base import SharedVLMEngine
from vlm.scene_understanding import SceneUnderstandingVLM


class SceneUnderstandingStage(PipelineStage):
    def __init__(
        self,
        shared_vlm: SharedVLMEngine | None = None,
        desc_path: Path = DESC_PATH,
        output_path: Path = PRED_JSON,
    ):
        self.shared_vlm = shared_vlm
        self.desc_path = desc_path
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    def run(self) -> None:
        if self.shared_vlm is None:
            raise RuntimeError("shared_vlm is required to run SceneUnderstandingStage")

        logger = logging.getLogger(__name__)
        logger.info("Scene understanding...")

        descriptions = load_descriptions(self.desc_path)
        selected_by_object = SelectCropsStage(desc_path=self.desc_path).load_output()
        vlm_task = SceneUnderstandingVLM(shared=self.shared_vlm)

        predictions = {}
        for obj_id, selected in selected_by_object.items():
            desc = descriptions[obj_id][0]
            logger.info("Querying task VLM for %s: %s (%d crops)", obj_id, desc, len(selected))
            try:
                response_text = vlm_task.query(selected, desc)
                predictions[f"id_{obj_id}"] = safe_json_list(response_text)
            except Exception:
                predictions[f"id_{obj_id}"] = []

        self.save(predictions)
        logger.info("Predictions saved to %s", self.output_path)

    def load_output(self) -> dict:
        with open(self.output_path, "r", encoding="utf-8") as f:
            return json.load(f)
