import json
import re
from typing import Any, List


def _extract_json_payload(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def safe_json_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = _extract_json_payload(str(raw).strip())
    if not text or text.lower() in ("none", "[]"):
        return []

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
            except Exception:
                return []

    if parsed is None:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip().lower() for x in parsed if str(x).strip()]
