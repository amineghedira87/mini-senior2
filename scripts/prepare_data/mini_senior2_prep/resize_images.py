
from pathlib import Path
from PIL import Image, ImageOps
from .config import *

def _make_448(in_path: Path, out_path: Path, size=448) -> bool:
    try:
        with Image.open(in_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            ratio = im.width / im.height
            # resize so that the smaller side becomes `size`, then center-crop to size×size
            if im.width < im.height:
                new_w = size; new_h = round(new_w / ratio)
            else:
                new_h = size; new_w = round(new_h * ratio)
            im = im.resize((int(new_w), int(new_h)), Image.BICUBIC)
            left = max(0, (im.width - size)//2)
            top  = max(0, (im.height - size)//2)
            im = im.crop((left, top, left + size, top + size))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, format="JPEG", quality=90, optimize=True)
            return True
    except Exception:
        return False

def _iter_originals(target: str):
    src = IMAGES_ORIG / target
    for p in sorted(src.glob("*")):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            yield p

def make_all_448():
    total_ok = total_fail = 0
    for t in TARGETS:
        (IMAGES_448 / t).mkdir(parents=True, exist_ok=True)
        ok = fail = 0
        for p in _iter_originals(t):
            out = IMAGES_448 / t / f"{p.stem}.jpg"
            if out.exists():
                ok += 1; continue
            if _make_448(p, out, 448):
                ok += 1
            else:
                fail += 1
        total_ok += ok; total_fail += fail
        print(f"{t}: 448 ok={ok}, fail={fail}")
    print(f"SUMMARY: 448 made={total_ok}, failed={total_fail}")

def make_some_448(limit_per_target=50):
    total_ok = total_fail = 0
    for t in TARGETS:
        (IMAGES_448 / t).mkdir(parents=True, exist_ok=True)
        ok = fail = 0
        count = 0
        for p in _iter_originals(t):
            out = IMAGES_448 / t / f"{p.stem}.jpg"
            if out.exists():
                ok += 1
            else:
                if _make_448(p, out, 448):
                    ok += 1
                else:
                    fail += 1
            count += 1
            if count >= limit_per_target:
                break
        total_ok += ok; total_fail += fail
        print(f"{t} (sample {limit_per_target}): 448 ok={ok}, fail={fail}")
    print(f"SUMMARY (sample): 448 made={total_ok}, failed={total_fail}")
