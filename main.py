import logging
import json
from pathlib import Path

from config import *
from evaluate.calculate_metrics import calculate_metrics
from evaluate.evaluator import Evaluator
from sam3.sam3_localization import SAM3Localizer
from support_objects.select_best_crops import select_best_crops_tournament
from support_objects.select_support_object import select_support_objects
from utils.aggregator import *
from utils.clear_memory import release_model
from utils.cropper import save_crop
from utils.data_loader import load_descriptions, load_frame_and_mask
from utils.gt_builder import GTBuilder
from utils.prediction_parser import safe_detailed_descriptions, safe_json_list
from vlm.base import SharedVLMEngine
from vlm.crop_selector import CropSelectorVLM
from vlm.gt_refinement import GTRefinementVLM
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

    shared_vlm = SharedVLMEngine.build(model_name=TASK_MODEL_NAME, gpu_memory_utilization=0.5)
    try:
        CROPS_DIR.mkdir(exist_ok=True)
        logger.info("Loading object descriptions...")
        descriptions = load_descriptions(DESC_PATH)

        frame_names = sorted([f.name for f in FRAMES_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")])
        logger.info(f"Processing {len(frame_names)} frames...")
        """
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
        """
        with open(TEMP_GT_JSON, "r", encoding="utf-8") as f:
            temp_gt = json.load(f)
            temp_gt = {int(k) if k.isdigit() else k: v for k, v in temp_gt.items()}
        try:
            with open(SELECTED_CROPS, "r", encoding="utf-8") as f:
                selected_crops_cache = json.load(f)
        except FileNotFoundError:
            selected_crops_cache = {}

        object_crops = collect_crops_by_object(CROPS_DIR)
        selected_by_object = {}
        final_result = {}
        detailed_result = {}
        final_gt = {}

        logger.info("Stage 1/4: selecting best crops...")
        vlm_selector = CropSelectorVLM(shared=shared_vlm)
        for obj_id, crop_paths in object_crops.items():
            desc = descriptions[obj_id][0]
            cache_key = str(obj_id)
            if cache_key in selected_crops_cache:
                selected = [Path(p) for p in selected_crops_cache[cache_key]]
            else:
                try:
                    selected_paths = select_best_crops_tournament(crop_paths, vlm_selector, desc, obj_id)
                except Exception:
                    selected_paths = []

                selected = selected_paths
                selected_crops_cache[cache_key] = [str(p) for p in selected_paths]
                save_result(selected_crops_cache, SELECTED_CROPS)
            selected_by_object[obj_id] = selected

        
        logger.info("Stage 2/4: scene understanding...")
        vlm_task = SceneUnderstandingVLM(shared=shared_vlm)
        for obj_id, selected in selected_by_object.items():
            desc = descriptions[obj_id][0]
            logger.info(f"Querying task VLM for {obj_id}: {desc} ({len(selected)} crops)")
            try:
                response_text = vlm_task.query(selected, desc)
                final_result[f"id_{obj_id}"] = safe_json_list(response_text)
            except Exception:
                final_result[f"id_{obj_id}"] = []
        save_result(final_result, PRED_JSON)

        with open(PRED_JSON, "r", encoding="utf-8") as f:
            final_result = json.load(f)

        logger.info("Stage 3/4: detailed item descriptions...")
        vlm_detailer = ItemDetailerVLM(shared=shared_vlm)
        for obj_id, selected in selected_by_object.items():
            desc = descriptions[obj_id][0]
            logger.info(f"Querying detail VLM for {obj_id}: {desc} ({len(selected)} crops)")
            try:
                detailed = vlm_detailer.query(selected, desc, final_result[f"id_{obj_id}"])
                detailed_result[f"id_{obj_id}"] = safe_detailed_descriptions(detailed)
            except Exception:
                detailed_result[f"id_{obj_id}"] = []
        save_result(detailed_result, DETAILED_PRED_JSON)

        with open(DETAILED_PRED_JSON, "r", encoding="utf-8") as f:
            detailed_result = json.load(f)
        

        logger.info("Stage 4/4: SAM3 localization on original frames...")
        from vlm.mask_chooser import SAM3MaskChooserVLM

        mask_chooser = SAM3MaskChooserVLM(shared=shared_vlm)
        sam3_localizer = SAM3Localizer(mask_chooser_vlm=mask_chooser)
        for obj_id, selected in selected_by_object.items():
            try:
                id_key = f"id_{obj_id}"
                items = detailed_result.get(id_key) or final_result.get(id_key, [])
                sam3_localizer.localize_object(
                    obj_id=obj_id,
                    selected_crops=selected,
                    items=items,
                )
            except Exception as e:
                logger.exception(f"SAM3 localization failed: {e}")

        logger.info("Stage 4/4: GT refinement...")
        vlm_refiner = GTRefinementVLM(shared=shared_vlm)
        for obj_id, selected in selected_by_object.items():
            desc = descriptions[obj_id][0]
            candidates = temp_gt.get(obj_id, [])
            logger.info(f"Querying refiner VLM for {obj_id}: {desc} ({len(selected)} crops)")
            try:
                response_text = vlm_refiner.query(selected, desc, candidates)
                final_gt[f"id_{obj_id}"] = safe_json_list(response_text)
            except Exception:
                final_gt[f"id_{obj_id}"] = []
        save_result(final_gt, GT_JSON)

        with open(PRED_JSON, "r", encoding="utf-8") as f:
            final_result = json.load(f)
        with open(GT_JSON, "r", encoding="utf-8") as f:
            final_gt = json.load(f)

        logger.info("Running evaluation...")
        evaluator = Evaluator(descriptions)
        results = evaluator.evaluate(final_result, final_gt)
        save_result(results, REPORT_JSON)

        logger.info("Calculating metrics...")
        metrics = calculate_metrics(results)
        save_result(metrics, METRICS_JSON)

    finally:
        shared_vlm.shutdown()

if __name__ == "__main__":
    main()