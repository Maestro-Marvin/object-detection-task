import logging

from config import *
from sam3.localization_runner import SAM3LocalizationRunner
from vlm.base import SharedVLMEngine
from vlm.frame_consolidator import FrameConsolidator
from vlm.item_detailer import ItemDetailerVLM
from vlm.scene_understanding import SceneUnderstandingVLM

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
    logger = logging.getLogger(__name__)

    frame_paths = sorted([p for p in FRAMES_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")], key=lambda p: p.name)

    shared_vlm = SharedVLMEngine.build(model_name=TASK_MODEL_NAME, gpu_memory_utilization=0.3)
    try:
        logger.info(f"Processing {len(frame_paths)} frames (end-to-end mode)...")

        logger.info("Stage 1/4: support consolidation...")
        consolidation = FrameConsolidator(shared=shared_vlm).consolidate(frame_paths, persist=True)

        logger.info("Stage 2/4: associated items prediction...")
        vlm_task = SceneUnderstandingVLM(shared=shared_vlm)
        final_result = vlm_task.predict_associated_items(
            consolidation.selected_by_support,
            persist=True,
        )

        logger.info("Stage 3/4: detailed item descriptions...")
        vlm_detailer = ItemDetailerVLM(shared=shared_vlm)
        detailed_result = vlm_detailer.predict_detailed_descriptions(
            consolidation.selected_by_support,
            final_result,
            persist=True,
        )

        logger.info("Stage 4/4: localization on original frames...")
        SAM3LocalizationRunner(shared_vlm).localize_all(
            consolidation.selected_by_support,
            detailed_result,
            logger=logger,
        )
    finally:
        shared_vlm.shutdown()

if __name__ == "__main__":
    main()