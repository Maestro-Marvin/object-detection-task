from pathlib import Path
from config import TOURNAMENT_TARGET_CROPS
from vlm.crop_selector import CropSelectorVLM

def select_best_crops_tournament(
    crop_paths: list[Path],
    selector: CropSelectorVLM,
    description: str,
    obj_id: int,
) -> list[Path]:
    """
    Турнирное "прореживание" кропов: сравниваем попарно и оставляем лучший кадр,
    пока не останется ровно TOURNAMENT_TARGET_CROPS.
    Турнир можно прервать посередине раунда, если набралось ровно target кадров.
    """
    target = TOURNAMENT_TARGET_CROPS
    crops = list(crop_paths)
    if len(crops) <= target:
        return crops

    crops.sort(key=lambda p: p.name)

    while len(crops) > target:
        next_round: list[Path] = []
        i = 0
        while i < len(crops):
            if len(next_round) + (len(crops) - i) == target:
                return next_round + crops[i:]

            if i == len(crops) - 1:
                next_round.append(crops[i])
                break
            a, b = crops[i], crops[i + 1]
            try:
                winner = selector.query([a, b], description, obj_id)
                next_round.append(a if winner == "A" else b)
            except Exception:
                next_round.append(a)
            i += 2
        crops = next_round

    return crops
