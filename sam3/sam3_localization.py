from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
import torch

from .sam3_rendering import (
    LocalizationOutputs,
    ensure_output_dirs,
    make_stem,
    make_stem_in_obj_dir,
    sanitize_label,
    draw_masks_overlay,
    put_title,
    save_overlay,
    save_union_mask,
)

from config import FRAMES_DIR, LOCALIZATION_DIR, SAM3_MODEL_PATH, SAM3_CONF, SAM3_HALF, SAM3_SAVE_BINARY_MASKS
from config import MAX_CROPS_PER_REQUEST, SAM3_AGENT_TOPK
from vlm.mask_chooser import SAM3MaskChooserVLM

class SAM3Localizer:
    """
    Локализация объектов на ИСХОДНЫХ кадрах через SAM3 (Ultralytics).

    Использование в stage:
      localizer = SAM3Localizer(...)
      for obj_id, selected_crops in selected_by_object.items():
          localizer.localize_object(obj_id, selected_crops, labels)
    """

    def __init__(
        self,
        frames_dir: Path = FRAMES_DIR,
        out_dir: Path = LOCALIZATION_DIR,
        sam3_model_path: Path = SAM3_MODEL_PATH,
        conf: float = SAM3_CONF,
        half: bool = SAM3_HALF,
        save_binary_masks: bool = SAM3_SAVE_BINARY_MASKS,
        mask_chooser_vlm: Optional[SAM3MaskChooserVLM] = None,
    ):
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.frames_dir = frames_dir
        self.out_dir = out_dir
        self.sam3_model_path = sam3_model_path
        self.conf = conf
        self.half = half
        self.save_binary_masks = save_binary_masks
        self.mask_chooser_vlm = mask_chooser_vlm

        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=str(sam3_model_path),
            half=half,
            verbose=False,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)


    def localize_object(self, obj_id: int, selected_crops: List[Path], items: List[Any]) -> None:
        """
        Локализует все `items` (подробные описания или label) на исходных кадрах, соответствующих `selected_crops`.
        """
        id_key = f"id_{obj_id}"
        if not items:
            return

        # Корневая папка для конкретного опорного объекта: out_dir/id_<obj_id>/
        obj_root = self.out_dir / id_key

        for crop_path in selected_crops:
            frame_name = Path(crop_path).name
            src_path = self.frames_dir / frame_name

            im = cv2.imread(str(src_path))
            src_shape = im.shape[:2]

            self.predictor.set_image(str(src_path))

            for item in items:
                label_name = item["label"]
                text_prompt = item["description"]
                label_dir = obj_root / sanitize_label(label_name)

                masks_t, boxes_t = self.predictor.inference_features(
                    self.predictor.features, src_shape=src_shape, text=[label_name]
                )

                if masks_t is None or masks_t.shape[0] == 0:
                    continue

                if masks_t.ndim == 2:
                    masks_t = masks_t[None, ...]
     
                best_idx = 0
                # Если SAM3 вернул несколько инстансов и у нас есть MLLM-chooser,
                # попробуем выбрать правильный инстанс по детальному описанию.
                if self.mask_chooser_vlm is not None and masks_t.shape[0] > 1:
                    try:
                        # Отберём top-K по score, затем попросим MLLM выбрать один.
                        scores = boxes_t[:, 4]
                        topk = int(min(SAM3_AGENT_TOPK, MAX_CROPS_PER_REQUEST - 1, scores.shape[0]))
                        cand = torch.topk(scores, k=topk).indices.tolist()

                        # Сохраняем кандидатов прямо в папку label_dir.
                        label_dir.mkdir(parents=True, exist_ok=True)
                        overlay_paths: List[Path] = []
                        for j, idx in enumerate(cand, start=1):
                            m = masks_t[idx : idx + 1].detach().cpu().numpy()
                            overlay = draw_masks_overlay(im, m)
                            score = float(scores[idx].item())
                            out_path = label_dir / f"{Path(frame_name).stem}__cand_{j}__score_{score:.3f}.jpg"
                            cv2.imwrite(str(out_path), overlay)
                            overlay_paths.append(out_path)
                        chosen = self.mask_chooser_vlm.choose_best(
                            raw_image_path=src_path, overlay_paths=overlay_paths, item=item
                        )
                        if chosen is not None and 0 <= chosen < len(cand):
                            best_idx = int(cand[chosen])
                    except Exception:
                        pass

                masks = masks_t[best_idx : best_idx + 1].detach().cpu().numpy()

                stem = make_stem_in_obj_dir(frame_name=frame_name)
                item_outputs = ensure_output_dirs(label_dir)

                overlay = draw_masks_overlay(im, masks)
                overlay = put_title(overlay, text_prompt)
                save_overlay(item_outputs, stem, overlay)
                if self.save_binary_masks:
                    save_union_mask(item_outputs, stem, masks)

