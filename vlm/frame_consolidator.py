from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from config import (
    FRAME_STRIDE,
    FRAMES_BY_SUPPORT_JSON,
    FRAMES_BY_SUPPORT_RAW_JSON,
    SELECTED_CROPS,
    SELECTOR_MODEL_NAME,
    TASK_MODEL_NAME,
)
from support_objects.select_best_crops import select_best_crops_tournament
from utils.aggregator import save_result

from .base import SharedVLMEngine, VLMClient


def _sorted_support_dict(paths_by_support: dict[str, list[Path]]) -> dict[str, list[str]]:
    """Стабильный порядок ключей и имён файлов для JSON."""
    out = {k: sorted(p.name for p in paths) for k, paths in paths_by_support.items()}
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


class _SupportGrouperVLM(VLMClient):
    """
    Для одного кадра: есть ли доминирующий опорный объект и его метка (JSON present/label).
    """

    def __init__(self, model_name: str = TASK_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)
        self.sampling_params.max_tokens = 128

    def query(self, image_paths: List[Path]) -> str:
        prompt_text = """You are given ONE image frame from a household scene.

Task:
Decide whether there is a SINGLE dominant support object in the frame (a main surface/container/furniture),
and if yes, name it.

Support object examples:
desk, table, countertop, shelf, cabinet, bed, sofa, chair, sink, drawer, nightstand

Rules:
- If a clear single dominant support exists: present=true and label=<one generic lowercase singular support label>.
- If NO clear dominant support exists: present=false and label=null.
- Never output small movable items (book, bottle, keyboard, etc.).
- Use generic names, not brands.

Output format (STRICT JSON only):
{"present": true, "label": "desk"}
or
{"present": false, "label": null}
"""
        return self._run_inference(image_paths, prompt_text)

    @staticmethod
    def _parse_support_reply(raw: Any) -> Tuple[bool, Optional[str]]:
        if raw is None:
            return (False, None)

        text = str(raw).strip()
        if not text:
            return (False, None)

        try:
            parsed = json.loads(text)
        except Exception:
            return (False, None)

        if not isinstance(parsed, dict):
            return (False, None)

        present = bool(parsed.get("present", False))
        label = parsed.get("label", None)
        if not present:
            return (False, None)
        if label is None:
            return (True, None)
        if isinstance(label, str):
            s = label.strip().lower()
            return (True, s or None)
        return (True, None)

    def classify_frame(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        try:
            raw = self.query([image_path])
            return self._parse_support_reply(raw)
        except Exception:
            return (False, None)


class _FrameSelectorVLM(VLMClient):
    """Турнирный сравнитель двух кадров (A/B) для одной опоры."""

    def __init__(self, model_name: str = SELECTOR_MODEL_NAME, shared: Optional[SharedVLMEngine] = None):
        super().__init__(model_name, shared=shared)

    def query(self, image_paths: List[Path], description: str, obj_id: int) -> str:
        support_hint = str(description).strip().lower()
        support_line = f"Target support object: '{support_hint}'.\n" if support_hint else ""

        prompt_text = f"""You are selecting the best frame for a downstream pipeline.
You are shown two frames (A then B) from the same scene.

{support_line}
Goal:
Pick the frame that is BEST for understanding the target support object (if specified) and the items associated with it.

What "best" means (priority order):
1) The target support object is clearly visible (or, if not specified, ONE dominant support is clearly visible).
2) Items on/inside/near that support object are also visible and not heavily occluded.
3) Frame is sharp (low motion blur), well-focused, good exposure.
4) Camera is not too close / not too far; enough context around the support object.

Avoid frames where:
- multiple competing supports dominate equally (confusing),
- the main support is heavily occluded,
- strong motion blur / darkness.

Output format:
Return exactly one character: A or B. No other text."""

        out = self._run_inference(image_paths, prompt_text).strip().upper()
        if "A" in out and "B" not in out:
            return "A"
        if "B" in out and "A" not in out:
            return "B"
        if out.startswith("B"):
            return "B"
        return "A"


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
    Шаг 1a: прореживание (stride).
    Шаги 1b–1c: группировка по опоре (_SupportGrouperVLM) + турнир (_FrameSelectorVLM).
    """

    def __init__(self, shared: SharedVLMEngine):
        self._grouper = _SupportGrouperVLM(shared=shared)
        self._selector = _FrameSelectorVLM(shared=shared)

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
            if support_labels:
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
