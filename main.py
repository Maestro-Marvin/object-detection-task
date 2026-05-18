import logging

from config import TASK_MODEL_NAME
from pipeline.evaluation import EvaluationStage
from pipeline.gt_refinement import GtRefinementStage
from pipeline.scene_understanding import SceneUnderstandingStage
from pipeline.select_crops import SelectCropsStage
from pipeline.temp_gt import TempGtStage
from vlm.base import SharedVLMEngine


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()

    shared_vlm = SharedVLMEngine.build(model_name=TASK_MODEL_NAME, gpu_memory_utilization=0.5)
    try:
        TempGtStage().run()
        SelectCropsStage(shared_vlm).run()
        SceneUnderstandingStage(shared_vlm).run()
        GtRefinementStage(shared_vlm).run()
        EvaluationStage().run()
    finally:
        shared_vlm.shutdown()

if __name__ == "__main__":
    main()
