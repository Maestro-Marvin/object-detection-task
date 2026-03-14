import math
import logging
import json
import torch
import gc
from pathlib import Path
from config import *
from utils.data_loader import load_descriptions, load_frame_and_mask
from support_objects.select_support_object import select_support_objects
from utils.cropper import save_crop
from vlm.scene_understanding import SceneUnderstandingVLM
from vlm.gt_refinement import GTRefinementVLM
from evaluate.evaluator import Evaluator
from evaluate.calculate_metrics import calculate_metrics
from utils.aggregator import *
from utils.gt_builder import GTBuilder
from support_objects.select_best_crops import select_best_crops_tournament
from vlm.crop_selector import CropSelectorVLM

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
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
        if mask is None:
            continue
        supports = select_support_objects(mask, descriptions)
        support_ids = [obj["id"] for obj in supports]

        frame_id = frame_name.split(".")[0]
        for obj in supports:
            save_crop(rgb, mask, obj["bbox"], obj["id"], support_ids, frame_id, CROPS_DIR)

        gt_builder.process_frame(mask, supports)
    temp_gt = gt_builder.build_gt()
    save_result(temp_gt, TEMP_GT_JSON)
    
    with open(TEMP_GT_JSON, "r", encoding="utf-8") as f:
        temp_gt = json.load(f)
        temp_gt = {int(k) if k.isdigit() else k: v for k, v in temp_gt.items()}

    try:
        with open(SELECTED_CROPS, "r", encoding="utf-8") as f:
            selected_crops_cache = json.load(f)
    except FileNotFoundError:
        selected_crops_cache = {}

    logger.info("Initializing VLMs...")
    vlm_selector = CropSelectorVLM()
    vlm_task = SceneUnderstandingVLM()
    vlm_refiner = GTRefinementVLM()
    object_crops = collect_crops_by_object(CROPS_DIR)
    final_result = {}
    final_gt = {}

    for obj_id, crop_paths in object_crops.items():
        desc = descriptions[obj_id][0]

        cache_key = str(obj_id)
        if cache_key in selected_crops_cache:
            selected = [Path(p) for p in selected_crops_cache[cache_key]]
        else:
            try:
                selected_paths = select_best_crops_tournament(crop_paths, vlm_selector, desc, obj_id)
            except Exception:
                selected_paths = get_uniform_crops(crop_paths)

            selected = selected_paths
            selected_crops_cache[cache_key] = [str(p) for p in selected_paths]
            save_result(selected_crops_cache, SELECTED_CROPS)

        candidates = temp_gt.get(obj_id, [])
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
    
    
    save_result(final_gt, GT_JSON)
    save_result(final_result, PRED_JSON)

    del vlm_task
    del vlm_refiner
    del vlm_selector
    gc.collect()
    torch.cuda.empty_cache()

    with open(PRED_JSON, "r", encoding="utf-8") as f:
        final_result = json.load(f)
        final_result = {int(k) if k.isdigit() else k: v for k, v in final_result.items()}
    with open(GT_JSON, "r", encoding="utf-8") as f:
        final_gt = json.load(f)
        final_gt = {int(k) if k.isdigit() else k: v for k, v in final_gt.items()}

    logger.info("Running evaluation...")
    evaluator = Evaluator(descriptions)
    results = evaluator.evaluate(final_result, final_gt)
    save_result(results, REPORT_JSON)

    logger.info("Calculating metrics...")
    metrics = calculate_metrics(results)
    save_result(metrics, METRICS_JSON)

if __name__ == "__main__":
    main()