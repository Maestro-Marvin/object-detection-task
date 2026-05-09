from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from vlm.base import SharedVLMEngine

from .mask_chooser_vlm import SAM3MaskChooserVLM
from .sam3_localization import SAM3Localizer


class SAM3LocalizationRunner:
    """
    Этап локализации: SAM3Localizer + SAM3MaskChooserVLM (общий shared engine).
    Прогон по всем объектам из consolidation и detailed predictions.
    """

    def __init__(
        self,
        shared_vlm: SharedVLMEngine,
        *,
        localizer: Optional[SAM3Localizer] = None,
    ):
        if localizer is None:
            mask_chooser = SAM3MaskChooserVLM(shared=shared_vlm)
            localizer = SAM3Localizer(mask_chooser_vlm=mask_chooser)
        self._localizer = localizer

    def localize_all(
        self,
        selected_by_object: Dict[int, List[Path]],
        detailed_result: Dict[str, List[Any]],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        log = logger or logging.getLogger(__name__)
        for obj_id, selected in selected_by_object.items():
            try:
                items = detailed_result[f"id_{obj_id}"]
                self._localizer.localize_object(
                    obj_id=obj_id,
                    selected_crops=selected,
                    items=items,
                )
            except Exception as e:
                log.exception(f"SAM3 localization failed: {e}")
