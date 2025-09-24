# mini_senior2_prep/ocr_images.py
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

try:
    import easyocr  # CPU
except ModuleNotFoundError as e:
    raise RuntimeError("EasyOCR not installed. Run: pip install easyocr") from e

from .config import PREP, TARGETS, IMAGES_448
from .io_utils import write_json

OCR_DIR = PREP / "ocr"
OCR_DIR.mkdir(parents=True, exist_ok=True)

# -------- EasyOCR singleton ----------
_READER = None
def _get_reader():
    """Create/reuse a single EasyOCR Reader (CPU)."""
    global _READER
    if _READER is None:
        _READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _READER

# -------- Core OCR ----------
def ocr_image(image_path: Path) -> List[Dict[str, Any]]:
    """Run OCR on one image and return a list of {text, conf, box} (may be empty)."""
    reader = _get_reader()
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    results = reader.readtext(arr, detail=1)  # [(box, text, conf), ...]
    lines: List[Dict[str, Any]] = []
    for box, text, conf in results:
        t = (text or "").strip()
        if t:
            try:
                c = float(conf)
            except Exception:
                c = None
            lines.append({"text": t, "conf": c, "box": box})
    return lines

def _cache_path(target: str, post_index: str) -> Path:
    return OCR_DIR / target / f"{post_index}.json"

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_cache(target: str, post_index: str, image_path: Path, lines: List[Dict[str, Any]]):
    """Write (or overwrite) cache JSON. Always writes—even if lines == []."""
    out = _cache_path(target, post_index)
    _ensure_dir(out)
    write_json(out, {
        "post_index": post_index,
        "image": str(image_path),
        "lines": lines
    })

# -------- Target-level OCR with retries ----------
def ocr_target(
    target: str,
    limit: Optional[int] = None,
    retries: int = 3,
    backoff_sec: float = 1.0,
    overwrite: bool = False,
    verbose: bool = True
) -> Dict[str, int]:
    """Run OCR over 448x images for a target and cache results.

    Caches are written to:
        prepared/ocr/{target}/{post_index}.json

    Behavior:
      - Writes a cache file EVEN IF no text is found (prevents reprocessing).
      - Retries transient failures (HTTP/decoder/etc.).
      - If overwrite=False, existing caches are skipped.

    Returns:
      dict with counts: {"done", "skip", "err", "sec"}
    """
    img_dir = IMAGES_448 / target
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if limit is not None:
        imgs = imgs[:limit]

    done = skip = err = 0
    t0 = time.time()

    for p in imgs:
        pid = p.stem
        cache = _cache_path(target, pid)

        if cache.exists() and not overwrite:
            skip += 1
            continue

        attempt = 0
        success = False
        last_exc = None
        while attempt <= retries:
            attempt += 1
            try:
                lines = ocr_image(p)  # may be empty
                _write_cache(target, pid, p, lines)
                success = True
                break
            except Exception as e:
                last_exc = e
                if attempt <= retries:
                    time.sleep(backoff_sec)
                else:
                    break

        if success:
            done += 1
            if verbose and done % 200 == 0:
                print(f"{target}: processed {done} images...")
        else:
            err += 1
            if verbose:
                print(f"[ERR] {target} | post_index={pid} | {type(last_exc).__name__}: {last_exc}")

    return {"done": done, "skip": skip, "err": err, "sec": int(time.time() - t0)}

def ocr_all_targets(limit_per_target: Optional[int] = None, **kwargs) -> Dict[str, Dict[str, int]]:
    """Run ocr_target() for each target; pass through kwargs (retries, overwrite, etc.)."""
    stats = {}
    for t in TARGETS:
        stats[t] = ocr_target(t, limit=limit_per_target, **kwargs)
    return stats
