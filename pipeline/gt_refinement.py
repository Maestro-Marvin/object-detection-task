import json
import logging
from pathlib import Path

from config import DESC_PATH, GT_JSON
from pipeline.base import PipelineStage
from pipeline.select_crops import SelectCropsStage
from pipeline.temp_gt import TempGtStage
from utils.data_loader import load_descriptions
from utils.prediction_parser import safe_json_list
from vlm.base import SharedVLMEngine
from vlm.gt_refinement import GTRefinementVLM


class GtRefinementStage(PipelineStage):
    def __init__(
        self,
        shared_vlm: SharedVLMEngine | None = None,
        desc_path: Path = DESC_PATH,
        output_path: Path = GT_JSON,
    ):
        self.shared_vlm = shared_vlm
        self.desc_path = desc_path
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    def run(self) -> None:
        if self.shared_vlm is None:
            raise RuntimeError("shared_vlm is required to run GtRefinementStage")

        logger = logging.getLogger(__name__)
        logger.info("GT refinement...")

        descriptions = load_descriptions(self.desc_path)
        selected_by_object = SelectCropsStage(desc_path=self.desc_path).load_output()
        temp_gt = TempGtStage(desc_path=self.desc_path).load_output()
        vlm_refiner = GTRefinementVLM(shared=self.shared_vlm)

        final_gt = {}
        for obj_id, selected in selected_by_object.items():
            desc = descriptions[obj_id][0]
            candidates = temp_gt.get(obj_id, [])
            logger.info(
                "Querying refiner VLM for %s: %s (%d crops)", obj_id, desc, len(selected)
            )
            try:
                response_text = vlm_refiner.query(selected, desc, candidates)
                parsed = safe_json_list(response_text)
                if not parsed and response_text.strip() and candidates:
                    logger.warning(
                        "Empty parse for object %s, raw response: %r", obj_id, response_text[:500]
                    )
                final_gt[f"id_{obj_id}"] = parsed
            except Exception:
                logger.exception("GT refinement failed for object %s", obj_id)
                final_gt[f"id_{obj_id}"] = []

        self.save(final_gt)
        logger.info("Ground truth saved to %s", self.output_path)

    def load_output(self) -> dict:
        with open(self.output_path, "r", encoding="utf-8") as f:
            return json.load(f)
