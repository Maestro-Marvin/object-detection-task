import re
from typing import Dict, List, Tuple


_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_REL_RE = re.compile(r"\]\s*(on|inside|near)\s+", flags=re.IGNORECASE)


def extract_items_with_relations(prediction_text: str) -> List[Tuple[str, str]]:
    """
    Извлекает элементы и их пространственное отношение из строки предсказания.

    Ожидаемый формат (пример):
      "[item1, item2] on desk id=15; [item3] inside desk id=15; [item4] near desk id=15"

    Возвращает список пар (item, relation), где relation ∈ {"on","inside","near"}.
    Если relation не удаётся определить для конкретных скобок — используется "unknown".
    """
    if not prediction_text:
        return []

    text = prediction_text.strip()
    if text.lower() == "none":
        return []

    out: List[Tuple[str, str]] = []

    # Идём по всем "[...]" и пытаемся взять relation сразу после закрывающей скобки.
    for m in _BRACKET_RE.finditer(text):
        items_blob = m.group(1)
        after = text[m.end() : m.end() + 32]
        rel_m = _REL_RE.search("]" + after)  # добавляем ']' чтобы regex совпал с шаблоном
        relation = rel_m.group(1).lower() if rel_m else "unknown"

        for raw in items_blob.split(","):
            item = raw.strip().lower()
            if item:
                out.append((item, relation))

    return out


def group_items_by_name(items_with_rel: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Превращает список (item, relation) в словарь item -> [relations].
    """
    grouped: Dict[str, List[str]] = {}
    for item, rel in items_with_rel:
        grouped.setdefault(item, [])
        if rel not in grouped[item]:
            grouped[item].append(rel)
    return grouped

