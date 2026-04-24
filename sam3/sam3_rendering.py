from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalizationOutputs:
    overlays_dir: Path
    masks_dir: Path


def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.strip().lower())


def ensure_output_dirs(out_dir: Path) -> LocalizationOutputs:
    overlays_dir = out_dir / "overlays"
    masks_dir = out_dir / "masks"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return LocalizationOutputs(overlays_dir=overlays_dir, masks_dir=masks_dir)


def make_stem(frame_name: str, id_key: str, text_prompt: str) -> str:
    safe_label = sanitize_label(text_prompt)
    return f"{Path(frame_name).stem}__{id_key}__{safe_label}"


def make_stem_in_obj_dir(frame_name: str, text_prompt: str) -> str:
    """
    Stem для случая, когда `id_key` уже закодирован в имени папки (localization/id_<obj>/...).
    """
    safe_label = sanitize_label(text_prompt)
    return f"{Path(frame_name).stem}__{safe_label}"


def draw_masks_overlay(image_bgr: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Рисует маски (N,H,W) поверх исходного BGR изображения.
    """
    from ultralytics.utils.plotting import Annotator, colors  # lazy import

    overlay = image_bgr.copy()
    annotator = Annotator(overlay, pil=False)
    annotator.masks(masks, [colors(i, True) for i in range(len(masks))])
    return annotator.result()


def put_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    """
    Подпись в левом верхнем углу (двойной контур для читаемости).
    """
    out = image_bgr
    cv2.putText(out, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_overlay(outputs: LocalizationOutputs, stem: str, overlay_bgr: np.ndarray) -> Path:
    out_path = outputs.overlays_dir / f"{stem}.jpg"
    cv2.imwrite(str(out_path), overlay_bgr)
    return out_path


def save_union_mask(outputs: LocalizationOutputs, stem: str, masks: np.ndarray) -> Path:
    """
    Сохраняет объединённую бинарную маску (union по всем инстансам) в PNG.
    """
    union = (masks > 0).any(axis=0).astype(np.uint8) * 255
    out_path = outputs.masks_dir / f"{stem}.png"
    cv2.imwrite(str(out_path), union)
    return out_path

