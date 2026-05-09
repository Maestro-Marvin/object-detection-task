from __future__ import annotations

from pathlib import Path
from typing import List

from config import BEST_FRAMES_TOURNAMENT_ITERS
from support_objects.tournament import run_tournament_round
from vlm.frame_selector import FrameSelectorVLM


def select_best_frames_tournament(frame_paths: List[Path], selector: FrameSelectorVLM) -> List[Path]:
    """
    Турнирно отбирает лучшие кадры сцены.
    Использует тот же механизм, что и для кропов, но смысл сравнения другой.
    """
    frames = list(frame_paths)
    iters = max(0, int(BEST_FRAMES_TOURNAMENT_ITERS))
    for _ in range(iters):
        if len(frames) <= 1:
            break
        frames = run_tournament_round(
            frames,
            selector=selector,  # совместим по интерфейсу .query(A,B,...)->A/B
            description="global_frame_selection",
            obj_id=0,
        )
        if len(frames) == 0:
            break
    return frames

