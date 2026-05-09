from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import (
    FRAME_STRIDE,
    FRAMES_BY_SUPPORT_JSON,
    FRAMES_BY_SUPPORT_RAW_JSON,
    SELECTED_CROPS,
)
from support_objects.select_best_crops import select_best_crops_tournament
from utils.aggregator import save_result

from .base import SharedVLMEngine
from .frame_selector import FrameSelectorVLM
from .support_grouper import SupportGrouperVLM


def _sorted_support_dict(paths_by_support: dict[str, list[Path]]) -> dict[str, list[str]]:
    """Стабильный порядок ключей и имён файлов для JSON."""
    out = {k: sorted(p.name for p in paths) for k, paths in paths_by_support.items()}
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


@dataclass(frozen=True)
class FrameConsolidationResult:
    """Итог группировки по опорам и турнирного отбора кадров."""

    raw_frames_by_support: dict[str, list[str]]
    best_frames_by_support: dict[str, list[str]]
    best_paths_by_support: dict[str, list[Path]]
    support_labels: tuple[str, ...]  # без служебной группы __unknown__
    selected_by_object: dict[int, list[Path]]
    support_descriptions: dict[int, str]
    selected_crops_cache: dict[str, list[str]]


class FrameConsolidator:
    """
    Шаг 1a: прореживание последовательности (stride).
    Шаги 1b–1c: группировка по доминирующему опорному объекту (SupportGrouperVLM)
    и турнирный отбор лучших кадров (FrameSelectorVLM).
    """

    def __init__(self, shared: SharedVLMEngine):
        self._grouper = SupportGrouperVLM(shared=shared)
        self._selector = FrameSelectorVLM(shared=shared)

    def consolidate(
        self,
        frame_paths: list[Path],
        *,
        stride: Optional[int] = None,
        persist: bool = False,
        raw_json_path: Optional[Path] = None,
        best_json_path: Optional[Path] = None,
    ) -> FrameConsolidationResult:
        log = logging.getLogger(__name__)
        step = max(1, int(stride if stride is not None else FRAME_STRIDE))
        sampled = frame_paths[::step]
        log.info(f"Step 1a: sampled frames = {len(sampled)} (stride={step})")
        log.info("Steps 1b–1c: grouping by dominant support + tournament per group...")

        frames_by_support: dict[str, list[Path]] = {}
        for p in sampled:
            present, label = self._grouper.classify_frame(p)
            if not present:
                continue
            key = label if label is not None else "__unknown__"
            frames_by_support.setdefault(key, []).append(p)

        raw_frames_by_support = _sorted_support_dict(frames_by_support)

        best_paths_by_support: dict[str, list[Path]] = {}
        for support_label, paths in frames_by_support.items():
            selected = select_best_crops_tournament(
                crop_paths=paths,
                selector=self._selector,
                description=support_label,
                obj_id=0,
            )
            best_paths_by_support[support_label] = selected

        best_frames_by_support = _sorted_support_dict(best_paths_by_support)

        support_labels = tuple(k for k in sorted(best_paths_by_support.keys()) if k != "__unknown__")
        selected_by_object: dict[int, list[Path]] = {}
        support_descriptions: dict[int, str] = {}
        selected_crops_cache: dict[str, list[str]] = {}
        for obj_idx, support_label in enumerate(support_labels, start=1):
            selected = best_paths_by_support.get(support_label, [])
            selected_by_object[obj_idx] = selected
            support_descriptions[obj_idx] = support_label
            selected_crops_cache[str(obj_idx)] = [str(p) for p in selected]

        if persist:
            save_result(raw_frames_by_support, raw_json_path or FRAMES_BY_SUPPORT_RAW_JSON)
            save_result(best_frames_by_support, best_json_path or FRAMES_BY_SUPPORT_JSON)
            save_result(selected_crops_cache, SELECTED_CROPS)

        return FrameConsolidationResult(
            raw_frames_by_support=raw_frames_by_support,
            best_frames_by_support=best_frames_by_support,
            best_paths_by_support=best_paths_by_support,
            support_labels=support_labels,
            selected_by_object=selected_by_object,
            support_descriptions=support_descriptions,
            selected_crops_cache=selected_crops_cache,
        )
