import math
import logging
from pathlib import Path
from config import *
from utils.data_loader import load_descriptions, load_frame_and_mask
from support_objects.support_object_utils import select_support_objects
from utils.cropper import save_crop
from vlm.scene_understanding import SceneUnderstandingVLM
from vlm.gt_refinement import GTRefinementVLM
from evaluate.llm_evaluator import LlmEvaluator
from utils.aggregator import *
from utils.gt_builder import GTBuilder

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("pipeline.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def get_uniform_crops(crops : list[Path]):
    interval = math.ceil(len(crops) / MAX_CROPS_PER_REQUEST)
    if interval == 0:
        return crops
    return crops[::interval]


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    CROPS_DIR.mkdir(exist_ok=True)
    logger.info("Loading object descriptions...")
    descriptions = load_descriptions(DESC_PATH)

    frame_names = sorted([f.name for f in FRAMES_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")])
    logger.info(f"Processing {len(frame_names)} frames...")

    gt_builder = GTBuilder(descriptions)
    for frame_name in frame_names:
        logger.info(f"Processing {frame_name}...")
        rgb, mask = load_frame_and_mask(frame_name, FRAMES_DIR, MASKS_DIR)
        supports = select_support_objects(mask, descriptions)
        support_ids = [obj["id"] for obj in supports]

        frame_id = frame_name.split(".")[0]
        for obj in supports:
            save_crop(rgb, mask, obj["bbox"], obj["id"], support_ids, frame_id, CROPS_DIR)


        gt_builder.process_frame(mask, supports)
    temp_gt = gt_builder.build_gt()

    logger.info("Initializing VLMs...")
    vlm_task = SceneUnderstandingVLM()
    vlm_refiner = GTRefinementVLM()
    object_crops = collect_crops_by_object(CROPS_DIR)
    final_result = {}
    final_gt = {}

    for obj_id, crop_paths in object_crops.items():
        desc = descriptions.get(obj_id, f"object_{obj_id}")
        selected = get_uniform_crops(crop_paths)
        candidates = temp_gt[obj_id]
        logger.info(f"Querying task VLM for {obj_id}: {desc} ({len(selected)} crops)")

        try:
            response = vlm_task.query(selected, desc, obj_id)
            final_result[f"id_{obj_id}"] = response
        except Exception as e:
            final_result[f"id_{obj_id}"] = f"ERROR: {str(e)}"

        logger.info(f"Querying refiner VLM for {obj_id}: {desc} ({len(selected)} crops)")

        try:
            response = vlm_refiner.query(selected, desc, obj_id, candidates)
            final_gt[f"id_{obj_id}"] = response
        except Exception as e:
            final_gt[f"id_{obj_id}"] = f"ERROR: {str(e)}"
    
    save_result(final_result, PRED_JSON)
    save_result(final_gt, GT_JSON)

    logger.info("Running evaluation...")
    evaluator = LlmEvaluator()
    results = evaluator.evaluate_batch(final_gt, final_result)
    save_result(results, REPORT_JSON)

    logger.info("Calculating metrics...")
    metrics = evaluator.calculate_metrics(results)
    save_result(metrics, METRICS_JSON)

if __name__ == "__main__":
    main()