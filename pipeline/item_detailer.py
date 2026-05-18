import json
import logging
from pathlib import Path

from config import DESC_PATH, DETAILED_PRED_JSON
from pipeline.base import PipelineStage
from pipeline.scene_understanding import SceneUnderstandingStage
from pipeline.select_crops import SelectCropsStage
from utils.data_loader import load_descriptions
from utils.prediction_parser import safe_detailed_descriptions
from vlm.base import SharedVLMEngine
from vlm.item_detailer import ItemDetailerVLM


class ItemDetailerStage(PipelineStage):
    def __init__(
        self,
        shared_vlm: SharedVLMEngine | None = None,
        desc_path: Path = DESC_PATH,
        output_path: Path = DETAILED_PRED_JSON,
    ):
        self.shared_vlm = shared_vlm
        self.desc_path = desc_path
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    def run(self) -> None:
        if self.shared_vlm is None:
            raise RuntimeError("shared_vlm is required to run ItemDetailerStage")

        logger = logging.getLogger(__name__)
        logger.info("Detailed item descriptions...")

        descriptions = load_descriptions(self.desc_path)
        selected_by_object = SelectCropsStage(desc_path=self.desc_path).load_output()
        predictions = SceneUnderstandingStage(desc_path=self.desc_path).load_output()
        vlm_detailer = ItemDetailerVLM(shared=self.shared_vlm)

        detailed_result = {}
        for obj_id, selected in selected_by_object.items():
            desc = descriptions[obj_id][0]
            logger.info(
                "Querying detail VLM for %s: %s (%d crops)", obj_id, desc, len(selected)
            )
            try:
                detailed = vlm_detailer.query(selected, desc, predictions[f"id_{obj_id}"])
                detailed_result[f"id_{obj_id}"] = safe_detailed_descriptions(detailed)
            except Exception:
                detailed_result[f"id_{obj_id}"] = []

        self.save(detailed_result)
        logger.info("Detailed predictions saved to %s", self.output_path)

    def load_output(self) -> dict:
        with open(self.output_path, "r", encoding="utf-8") as f:
            return json.load(f)
