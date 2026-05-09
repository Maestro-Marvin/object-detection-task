from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, Sequence


class ABSelector(Protocol):
    def query(self, image_paths: List[Path], description: str, obj_id: int) -> str: ...


def run_tournament_round(
    paths: Sequence[Path],
    selector: ABSelector,
    description: str,
    obj_id: int,
) -> List[Path]:
    """
    Делает ОДИН раунд турнирного отбора (попарные сравнения A vs B).

    Важно: здесь НЕТ логики остановки по порогу — это просто один "reduce" раунд.
    """
    items = list(paths)
    if len(items) <= 1:
        return items

    items.sort(key=lambda p: p.name)

    next_round: List[Path] = []
    i = 0
    while i < len(items):
        if i == len(items) - 1:
            next_round.append(items[i])
            break
        a, b = items[i], items[i + 1]
        try:
            winner = selector.query([a, b], description, obj_id)
            winner = str(winner).strip().upper()
            if winner not in ("A", "B"):
                winner = "A"
            next_round.append(a if winner == "A" else b)
        except Exception:
            next_round.append(a)
        i += 2

    return next_round

