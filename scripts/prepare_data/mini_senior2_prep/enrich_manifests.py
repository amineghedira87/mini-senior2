
from pathlib import Path
import json, shutil
from typing import Iterable, Optional

from .config import MANIFESTS, PREP, TARGETS, SPLITS
from .io_utils import write_jsonl

OCR_DIR = PREP / "ocr"

def _load_ocr_lines(target: str, post_index: str):
    p = OCR_DIR / target / f"{post_index}.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    lines = []
    for seg in data.get("lines", []):
        t = seg.get("text")
        if isinstance(t, str) and t.strip():
            lines.append(t)
    return lines

def enrich_split_inplace_with_ocr(target: str, split: str, make_backup: bool = True) -> Path:
    """Add OCR strings directly into MANIFESTS/target/split.jsonl (in-place).
    If make_backup=True, writes a sibling backup once: split.backup.jsonl
    """
    inp = MANIFESTS / target / f"{split}.jsonl"
    if make_backup:
        bk = MANIFESTS / target / f"{split}.backup.jsonl"
        if not bk.exists():
            shutil.copy2(inp, bk)

    rows_out = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = str(row.get("post_index"))
            # always keep rows (even text-only)
            if "ocrs" not in row:
                row["ocrs"] = []
            if row.get("has_image") and row.get("image_path"):
                texts = _load_ocr_lines(target, pid)
                if texts:
                    # append (do not duplicate if rerun)
                    if not row["ocrs"]:
                        row["ocrs"] = [{"source": "easyocr@1.7.1", "text": t} for t in texts]
            rows_out.append(row)

    # overwrite original
    write_jsonl(inp, rows_out)
    return inp

def enrich_all_inplace_with_ocr(targets: Optional[Iterable[str]] = None, make_backup: bool = True):
    targets = list(targets) if targets else TARGETS
    outs = []
    for t in targets:
        for s in SPLITS:
            outs.append(str(enrich_split_inplace_with_ocr(t, s, make_backup=make_backup)))
    return outs
