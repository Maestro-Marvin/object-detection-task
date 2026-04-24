from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from .sam3_rendering import (
    LocalizationOutputs,
    ensure_output_dirs,
    make_stem,
    make_stem_in_obj_dir,
    draw_masks_overlay,
    put_title,
    save_overlay,
    save_union_mask,
)

from config import FRAMES_DIR, LOCALIZATION_DIR, SAM3_MODEL_PATH, SAM3_CONF, SAM3_HALF, SAM3_SAVE_BINARY_MASKS

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
    ):
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.frames_dir = frames_dir
        self.out_dir = out_dir
        self.sam3_model_path = sam3_model_path
        self.conf = conf
        self.half = half
        self.save_binary_masks = save_binary_masks

        # Выход структурируем по объектам: out_dir/id_<obj_id>/...
        # поэтому корневые overlays/masks не используем для записи.
        self._outputs = ensure_output_dirs(out_dir)

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

    def outputs(self) -> LocalizationOutputs:
        return self._outputs

    def localize_object(self, obj_id: int, selected_crops: List[Path], labels: List[str]) -> None:
        """
        Локализует все `labels` на всех исходных кадрах, соответствующих `selected_crops`.
        """
        id_key = f"id_{obj_id}"
        if not labels:
            return

        # Папка для конкретного опорного объекта
        obj_outputs = ensure_output_dirs(self.out_dir / id_key)

        for crop_path in selected_crops:
            frame_name = Path(crop_path).name
            src_path = self.frames_dir / frame_name

            im = cv2.imread(str(src_path))
            src_shape = im.shape[:2]

            self.predictor.set_image(str(src_path))

            for label in labels:
                text_prompt = str(label).strip()

                masks_t, _boxes_t = self.predictor.inference_features(
                    self.predictor.features, src_shape=src_shape, text=[text_prompt]
                )

                if masks_t is None:
                    continue

                masks = masks_t.cpu().numpy()
                if masks.ndim == 2:
                    masks = masks[None, ...]
                if masks.shape[0] == 0:
                    continue

                stem = make_stem(frame_name=frame_name, id_key=id_key, text_prompt=text_prompt)

                overlay = draw_masks_overlay(im, masks)
                overlay = put_title(overlay, f"{id_key} | {text_prompt}")
                save_overlay(obj_outputs, stem, overlay)
                if self.save_binary_masks:
                    save_union_mask(obj_outputs, stem, masks)

