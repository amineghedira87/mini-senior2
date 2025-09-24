
import json, csv
from pathlib import Path
from collections import defaultdict

from .config import PREP, MANIFESTS, TARGETS, SPLITS, IMAGES_448

REPORTS = PREP / "reports"
OCR_DIR  = PREP / "ocr"

def _has_image_item(row: dict) -> bool:
    return bool(row.get("has_image")) and bool(row.get("image_path"))

def _has_ocr_text(row: dict) -> bool:
    o = row.get("ocrs") or []
    for seg in o:
        t = seg.get("text") if isinstance(seg, dict) else None
        if isinstance(t, str) and t.strip():
            return True
    return False

def _cache_text_count(path: Path) -> int:
    """Return # of non-empty lines in an OCR cache json."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        lines = [seg.get("text","").strip() for seg in data.get("lines", []) if isinstance(seg, dict)]
        return sum(1 for x in lines if x)
    except Exception:
        return 0

def ocr_coverage_report():
    """
    Compute OCR coverage at item-level and unique-post level.
    Writes:
      - prepared/reports/ocr_coverage.json
      - prepared/reports/ocr_coverage.csv
    Returns dict(report).
    """
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ---------- item-level ----------
    item_rows = [("target","split","items_total","items_with_image","items_with_ocr_text",
                  "pct_with_image","pct_with_ocr_text","pct_ocr_given_image")]
    item_stats = {}

    for t in TARGETS:
        item_stats[t] = {}
        for s in SPLITS:
            fp = MANIFESTS / t / f"{s}.jsonl"
            n_total = n_img = n_ocr = 0
            if fp.exists():
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        n_total += 1
                        if _has_image_item(row):
                            n_img += 1
                            if _has_ocr_text(row):
                                n_ocr += 1
            pct_img = (n_img/n_total*100) if n_total else 0.0
            pct_ocr = (n_ocr/n_total*100) if n_total else 0.0
            pct_ocr_img = (n_ocr/n_img*100) if n_img else 0.0
            item_stats[t][s] = dict(
                items_total=n_total,
                items_with_image=n_img,
                items_with_ocr_text=n_ocr,
                pct_with_image=round(pct_img,2),
                pct_with_ocr_text=round(pct_ocr,2),
                pct_ocr_given_image=round(pct_ocr_img,2),
            )
            item_rows.append((t,s,n_total,n_img,n_ocr,
                              round(pct_img,2), round(pct_ocr,2), round(pct_ocr_img,2)))

    # ---------- unique-post level ----------
    post_rows = [("target","split","unique_posts_total","unique_posts_with_image",
                  "posts_with_ocr_cache","posts_with_ocr_text",
                  "pct_cache_given_image","pct_text_given_image")]
    post_stats = {}

    for t in TARGETS:
        post_stats[t] = {}
        # Pre-compute which posts have a cache and which have non-empty text
        caches = list((OCR_DIR/t).glob("*.json"))
        cache_set = set(p.stem for p in caches)
        cache_with_text = set(p.stem for p in caches if _cache_text_count(p) > 0)

        for s in SPLITS:
            fp = MANIFESTS / t / f"{s}.jsonl"
            posts_all = set()
            posts_with_img = set()
            if fp.exists():
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        pid = str(row.get("post_index"))
                        if pid:
                            posts_all.add(pid)
                            if _has_image_item(row):
                                posts_with_img.add(pid)

            posts_cache      = posts_with_img & cache_set
            posts_cache_text = posts_with_img & cache_with_text

            up_total = len(posts_all)
            up_img   = len(posts_with_img)
            up_cache = len(posts_cache)
            up_text  = len(posts_cache_text)

            pct_cache_img = (up_cache/up_img*100) if up_img else 0.0
            pct_text_img  = (up_text /up_img*100) if up_img else 0.0

            post_stats[t][s] = dict(
                unique_posts_total=up_total,
                unique_posts_with_image=up_img,
                posts_with_ocr_cache=up_cache,
                posts_with_ocr_text=up_text,
                pct_cache_given_image=round(pct_cache_img,2),
                pct_text_given_image=round(pct_text_img,2),
            )
            post_rows.append((t,s,up_total,up_img,up_cache,up_text,
                              round(pct_cache_img,2), round(pct_text_img,2)))

    # ---------- write CSV ----------
    csv_path = REPORTS / "ocr_coverage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# ITEM-LEVEL"])
        w.writerows(item_rows)
        w.writerow([])
        w.writerow(["# UNIQUE-POST LEVEL"])
        w.writerows(post_rows)

    # ---------- write JSON ----------
    json_path = REPORTS / "ocr_coverage.json"
    out = {"item_level": item_stats, "unique_post_level": post_stats}
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------- print a short summary ----------
    print("== OCR COVERAGE (item-level) ==")
    for t in TARGETS:
        for s in SPLITS:
            st = item_stats[t][s]
            print(f"{t.upper():7s} {s:5s} | total={st['items_total']:5d}  "
                  f"img={st['items_with_image']:5d} ({st['pct_with_image']:4.1f}%)  "
                  f"OCRtext={st['items_with_ocr_text']:5d} ({st['pct_with_ocr_text']:4.1f}%)  "
                  f"OCR|img={st['pct_ocr_given_image']:4.1f}%")

    print("\n== OCR COVERAGE (unique-post level) ==")
    for t in TARGETS:
        for s in SPLITS:
            st = post_stats[t][s]
            print(f"{t.upper():7s} {s:5s} | posts={st['unique_posts_total']:4d}  "
                  f"img_posts={st['unique_posts_with_image']:4d}  "
                  f"cache|img={st['pct_cache_given_image']:4.1f}%  "
                  f"text|img={st['pct_text_given_image']:4.1f}%")

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", json_path)
    return out
