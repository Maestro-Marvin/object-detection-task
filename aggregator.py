import json
from pathlib import Path
from collections import defaultdict

def collect_crops_by_object(crops_dir: Path) -> dict[int, list[Path]]:
    object_crops = defaultdict(list)
    for obj_dir in crops_dir.iterdir():
        if obj_dir.is_dir():
            try:
                obj_id = int(obj_dir.name)
                for crop in sorted(obj_dir.glob("*.jpg")):
                    object_crops[obj_id].append(crop)
            except ValueError:
                continue
    return object_crops

def save_final_result(result: dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)