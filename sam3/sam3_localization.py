from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, List, Optional

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

    def _build_text_prompt(self, item: Any) -> str:
        """
        Для SAM3 используем ТОЛЬКО простой label (без сложных описаний).
        """
        if isinstance(item, str):
            return item.strip()

        if not isinstance(item, dict):
            return str(item).strip()

        return str(item.get("label", "")).strip()

    def localize_object(self, obj_id: int, selected_crops: List[Path], items: List[Any]) -> None:
        """
        Локализует все `items` (подробные описания или label) на исходных кадрах, соответствующих `selected_crops`.
        """
        id_key = f"id_{obj_id}"
        if not items:
            return

        # Папка для конкретного опорного объекта
        obj_outputs = ensure_output_dirs(self.out_dir / id_key)

        for crop_path in selected_crops:
            frame_name = Path(crop_path).name
            src_path = self.frames_dir / frame_name

            im = cv2.imread(str(src_path))
            src_shape = im.shape[:2]

            self.predictor.set_image(str(src_path))

            for item in items:
                text_prompt = self._build_text_prompt(item)
                if not text_prompt:
                    continue

                masks_t, boxes_t = self.predictor.inference_features(
                    self.predictor.features, src_shape=src_shape, text=[text_prompt]
                )

                if masks_t is None:
                    continue

                if masks_t.ndim == 2:
                    masks_t = masks_t[None, ...]
                if masks_t.shape[0] == 0:
                    continue

                best_idx = 0
                chooser_selected = False
                # Если SAM3 вернул несколько инстансов и у нас есть MLLM-chooser,
                # попробуем выбрать правильный инстанс по детальному описанию.
                if self.mask_chooser_vlm is not None and masks_t.shape[0] > 1:
                    try:
                        # Отберём top-K по score (или по площади), затем попросим MLLM выбрать один.
                        if boxes_t is not None and boxes_t.ndim == 2 and boxes_t.shape[0] == masks_t.shape[0] and boxes_t.shape[1] >= 5:
                            scores = boxes_t[:, 4]
                            topk = int(min(SAM3_AGENT_TOPK, MAX_CROPS_PER_REQUEST - 1, scores.shape[0]))
                            cand = torch.topk(scores, k=topk).indices.tolist()
                        else:
                            areas = masks_t.to(dtype=torch.bool).sum(dim=(1, 2))
                            topk = int(min(SAM3_AGENT_TOPK, MAX_CROPS_PER_REQUEST - 1, areas.shape[0]))
                            cand = torch.topk(areas, k=topk).indices.tolist()

                        # Не сохраняем кандидатов в `localization/` — создаём временные файлы только для MLLM выбора.
                        overlay_paths: List[Path] = []
                        with tempfile.TemporaryDirectory(prefix="sam3_agent_") as td:
                            tmp_dir = Path(td)
                            for j, idx in enumerate(cand, start=1):
                                m = masks_t[idx : idx + 1].detach().cpu().numpy()
                                overlay = draw_masks_overlay(im, m)
                                overlay = put_title(overlay, f"{id_key} | {text_prompt} | cand {j}")
                                tmp_path = tmp_dir / f"cand_{j}.jpg"
                                cv2.imwrite(str(tmp_path), overlay)
                                overlay_paths.append(tmp_path)

                            chosen = self.mask_chooser_vlm.choose_best(
                                raw_image_path=src_path, overlay_paths=overlay_paths, item=item
                            )
                            if chosen is not None and 0 <= chosen < len(cand):
                                best_idx = int(cand[chosen])
                                chooser_selected = True
                    except Exception:
                        best_idx = 0

                # Фолбэк: max score (boxes_t[:, 4]) после NMS, иначе max area.
                if not chooser_selected:
                    try:
                        if boxes_t is not None and boxes_t.ndim == 2 and boxes_t.shape[0] > 0 and boxes_t.shape[1] >= 5:
                            best_idx = int(torch.argmax(boxes_t[:, 4]).item())
                        else:
                            areas = masks_t.to(dtype=torch.bool).sum(dim=(1, 2))
                            best_idx = int(torch.argmax(areas).item())
                    except Exception:
                        best_idx = 0

                masks = masks_t[best_idx : best_idx + 1].detach().cpu().numpy()

                stem = make_stem_in_obj_dir(frame_name=frame_name, text_prompt=text_prompt)

                overlay = draw_masks_overlay(im, masks)
                overlay = put_title(overlay, f"{id_key} | {text_prompt}")
                save_overlay(obj_outputs, stem, overlay)
                if self.save_binary_masks:
                    save_union_mask(obj_outputs, stem, masks)

