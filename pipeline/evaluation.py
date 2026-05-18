import json
import logging
from pathlib import Path

from config import DESC_PATH, METRICS_JSON, REPORT_JSON
from evaluate.calculate_metrics import calculate_metrics
from evaluate.evaluator import Evaluator
from pipeline.base import PipelineStage
from pipeline.gt_refinement import GtRefinementStage
from pipeline.scene_understanding import SceneUnderstandingStage
from utils.aggregator import save_result
from utils.data_loader import load_descriptions


class EvaluationStage(PipelineStage):
    def __init__(
        self,
        desc_path: Path = DESC_PATH,
        report_path: Path = REPORT_JSON,
        metrics_path: Path = METRICS_JSON,
    ):
        self.desc_path = desc_path
        self._output_path = report_path
        self.metrics_path = metrics_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    def run(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Running evaluation...")

        descriptions = load_descriptions(self.desc_path)
        predictions = SceneUnderstandingStage(desc_path=self.desc_path).load_output()
        ground_truth = GtRefinementStage(desc_path=self.desc_path).load_output()

        evaluator = Evaluator(descriptions)
        results = evaluator.evaluate(predictions, ground_truth)
        self.save(results)

        logger.info("Calculating metrics...")
        metrics = calculate_metrics(results)
        save_result(metrics, self.metrics_path)

        logger.info("Report saved to %s", self.output_path)
        logger.info("Metrics saved to %s", self.metrics_path)

    def load_output(self) -> list:
        with open(self.output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_metrics(self) -> dict:
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
