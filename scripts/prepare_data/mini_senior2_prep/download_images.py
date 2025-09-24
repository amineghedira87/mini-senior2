
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import requests
from .config import *
from .io_utils import load_json, write_json
from .reddit_image_resolver import candidates_for_post, smart_variants

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Referer": "https://www.reddit.com/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
})

def _is_image(resp):
    ct = (resp.headers.get("Content-Type") or "").lower()
    return resp.status_code == 200 and ct.startswith("image/")

def _download_with_variants(urls, dst_base: Path):
    tried = set()
    for u in urls:
        for v in smart_variants(u):
            if not v or v in tried:
                continue
            tried.add(v)
            try:
                r = session.get(v, stream=True, timeout=TIMEOUT_SEC, allow_redirects=True)
                if not _is_image(r):
                    continue
                ct = (r.headers.get("Content-Type") or "").lower()
                ext = ".png" if "png" in ct else ".jpg"
                dst = dst_base.with_suffix(ext)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk: f.write(chunk)
                if dst.stat().st_size > 0:
                    return dst
                dst.unlink(missing_ok=True)
            except Exception:
                pass
    return None

def _dl_one(target: str, pid: int, posts: dict):
    base = IMAGES_ORIG / target / str(pid)
    if base.with_suffix(".jpg").exists() or base.with_suffix(".png").exists():
        return True, None
    post_rec = posts.get(pid, {})
    cands = list(candidates_for_post(post_rec))
    if not cands:
        return False, "no_candidates"
    out = _download_with_variants(cands, base)
    return (out is not None), ("download_fail" if out is None else None)

def _gather_post_indices(target: str):
    # union of post_indices that appear in any split
    post_indices = set()
    for s in SPLITS:
        items = load_json(RAW / target / f"{s}.json")
        for it in items:
            post_indices.add(it["index"][0])
    return sorted(post_indices)

def download_some(limit_per_target=50):
    stats = {}
    total_ok = total_fail = 0
    for t in TARGETS:
        (IMAGES_ORIG / t).mkdir(parents=True, exist_ok=True)
        posts = load_json(RAW / t / "post.json")
        post_indices = _gather_post_indices(t)[:limit_per_target]

        ok = fail = 0
        reasons = defaultdict(int)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_dl_one, t, pid, posts): pid for pid in post_indices}
            for fut in as_completed(futs):
                success, reason = fut.result()
                if success: ok += 1
                else:
                    fail += 1
                    reasons[reason] += 1

        stats[t] = {"ok": ok, "fail": fail, "reasons": dict(reasons)}
        total_ok += ok; total_fail += fail
        print(f"{t} (sample): ok={ok}  fail={fail}  reasons={dict(reasons)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / "image_download_report_sample.json", stats)
    print(f"\nSUMMARY (sample): originals ok={total_ok}, fail={total_fail}")

def download_all():
    stats = {}
    total_ok = total_fail = 0
    for t in TARGETS:
        (IMAGES_ORIG / t).mkdir(parents=True, exist_ok=True)
        posts = load_json(RAW / t / "post.json")

        post_indices = _gather_post_indices(t)
        ok = fail = 0
        reasons = defaultdict(int)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_dl_one, t, pid, posts): pid for pid in post_indices}
            for fut in as_completed(futs):
                success, reason = fut.result()
                if success: ok += 1
                else:
                    fail += 1
                    reasons[reason] += 1

        stats[t] = {"ok": ok, "fail": fail, "reasons": dict(reasons)}
        total_ok += ok; total_fail += fail
        print(f"{t}: ok={ok}  fail={fail}  reasons={dict(reasons)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / "image_download_report.json", stats)
    print(f"\nSUMMARY: originals ok={total_ok}, fail={total_fail}")
