from pathlib import Path
from config import MAX_CROPS_PER_REQUEST
from vlm.crop_selector import CropSelectorVLM
from support_objects.tournament import run_tournament_round

def select_best_crops_tournament(
    crop_paths: list[Path],
    selector: CropSelectorVLM,
    description: str,
    obj_id: int,
) -> list[Path]:
    """
    Турнирное "прореживание" кропов: сравниваем попарно и оставляем лучший кадр,
    пока не останется <= MAX_CROPS_PER_REQUEST.
    """
    crops = list(crop_paths)
    if len(crops) <= MAX_CROPS_PER_REQUEST:
        return crops

    while len(crops) > MAX_CROPS_PER_REQUEST:
        crops = run_tournament_round(crops, selector=selector, description=description, obj_id=obj_id)

    return crops