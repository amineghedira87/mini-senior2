
import json
from pathlib import Path

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
