import logging
import json
from pathlib import Path

from config import *
from sam3.sam3_localization import SAM3Localizer
from support_objects.select_best_crops import select_best_crops_tournament
from utils.aggregator import *
from utils.prediction_parser import safe_detailed_descriptions, safe_json_list, safe_support_group
from vlm.base import SharedVLMEngine
from vlm.frame_selector import FrameSelectorVLM
from vlm.item_detailer import ItemDetailerVLM
from vlm.mask_chooser import SAM3MaskChooserVLM
from vlm.scene_understanding import SceneUnderstandingVLM
from vlm.support_grouper import SupportGrouperVLM

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

        # Step 1a: keep only each N-th frame
        sampled_frames = frame_paths[:: max(1, int(FRAME_STRIDE))]
        logger.info(f"Step 1a: sampled frames = {len(sampled_frames)} (stride={FRAME_STRIDE})")

        # Step 1b: group sampled frames by dominant support (VLM)
        logger.info("Step 1b: grouping sampled frames by dominant support...")
        grouper = SupportGrouperVLM(shared=shared_vlm)
        frames_by_support: dict[str, list[Path]] = {}
        for p in sampled_frames:
            try:
                raw = grouper.query([p])
                present, label = safe_support_group(raw)
            except Exception:
                present, label = (False, None)
            if not present:
                continue
            key = label if label is not None else "__unknown__"
            frames_by_support.setdefault(key, []).append(p)

        # Save raw grouping before tournament (for debugging)
        raw_frames_by_support: dict[str, list[str]] = {}
        for k, paths in frames_by_support.items():
            raw_frames_by_support[k] = sorted([p.name for p in paths])
        raw_frames_by_support = dict(sorted(raw_frames_by_support.items(), key=lambda kv: kv[0]))
        save_result(raw_frames_by_support, Path("results/frames_by_support_raw.json"))

        # Step 1c: tournament selection inside each support group until <= 5 frames
        logger.info("Step 1c: tournament selection per support group...")
        frame_selector = FrameSelectorVLM(shared=shared_vlm)
        best_frames_by_support: dict[str, list[str]] = {}
        best_paths_by_support: dict[str, list[Path]] = {}
        for support_label, paths in frames_by_support.items():
            selected = select_best_crops_tournament(
                crop_paths=paths,
                selector=frame_selector,
                description=support_label,
                obj_id=0,
            )
            best_paths_by_support[support_label] = selected
            best_frames_by_support[support_label] = sorted([p.name for p in selected])

        # стабильный порядок для удобного диффа/чтения
        best_frames_by_support = dict(sorted(best_frames_by_support.items(), key=lambda kv: kv[0]))
        save_result(best_frames_by_support, Path("results/frames_by_support.json"))

        # Далее работаем по группам: support_label -> выбранные кадры (<=5)
        support_labels = [k for k in sorted(best_paths_by_support.keys()) if k != "__unknown__"]
        if not support_labels:
            logger.warning("No support groups found (all frames unknown/absent). Stopping pipeline.")
            return

        selected_by_object: dict[int, list[Path]] = {}
        support_descriptions: dict[int, str] = {}
        selected_crops_cache: dict[str, list[str]] = {}

        for obj_idx, support_label in enumerate(support_labels, start=1):
            selected = best_paths_by_support.get(support_label, [])
            selected_by_object[obj_idx] = selected
            support_descriptions[obj_idx] = support_label
            selected_crops_cache[str(obj_idx)] = [str(p) for p in selected]

        save_result(selected_crops_cache, SELECTED_CROPS)

        logger.info("Stage 2/4: scene understanding...")
        vlm_task = SceneUnderstandingVLM(shared=shared_vlm)
        final_result = {}
        for obj_id, selected in selected_by_object.items():
            desc = support_descriptions[obj_id]
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
        detailed_result = {}
        for obj_id, selected in selected_by_object.items():
            desc = support_descriptions[obj_id]
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

        mask_chooser = SAM3MaskChooserVLM(shared=shared_vlm)
        sam3_localizer = SAM3Localizer(mask_chooser_vlm=mask_chooser)
        for obj_id, selected in selected_by_object.items():
            try:
                items = detailed_result[f"id_{obj_id}"]
                sam3_localizer.localize_object(
                    obj_id=obj_id,
                    selected_crops=selected,
                    items=items,
                )
            except Exception as e:
                logger.exception(f"SAM3 localization failed: {e}")
    finally:
        shared_vlm.shutdown()

if __name__ == "__main__":
    main()