import re
import json
from typing import Any, Dict, List, Tuple, Optional


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


def safe_json_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in ("none", "[]"):
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip().lower() for x in parsed if isinstance(x, str) and x.strip()]


def safe_json_string_or_null(raw: Any) -> Optional[str]:
    """
    Парсит строго JSON-строку (например: "desk") или null.
    Возвращает нормализованную строку (lower/strip) или None.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
    else:
        text = str(raw).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        # Если модель вернула просто слово без кавычек — примем как строку.
        return text.strip().lower() if text else None
    if parsed is None:
        return None
    if isinstance(parsed, str):
        s = parsed.strip().lower()
        return s or None
    return None


def safe_detailed_descriptions(raw: Any) -> List[dict]:
    if isinstance(raw, list):
        valid_items = []
        for item in raw:
            if isinstance(item, dict) and "label" in item:
                valid_items.append(item)
        return valid_items
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in ("none", "[]"):
        return []
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text, flags=re.IGNORECASE)
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            valid_items = []
            for item in parsed:
                if isinstance(item, dict) and "label" in item:
                    valid_items.append(item)
            return valid_items
    except Exception:
        pass
    return []


def safe_support_group(raw: Any) -> Tuple[bool, Optional[str]]:
    """
    Парсит JSON вида {"present": true/false, "label": "desk"|null}.
    Возвращает (present, label).
    """
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