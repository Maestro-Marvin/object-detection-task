import json
from typing import Any, List


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
