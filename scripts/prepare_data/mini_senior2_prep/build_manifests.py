from pathlib import Path
from .config import RAW, TARGETS, SPLITS, IMAGES_448, MANIFESTS
from .io_utils import load_json, write_jsonl

def _img_448_exists(target: str, post_index) -> tuple[bool, str|None]:
    jpg = IMAGES_448 / target / f"{post_index}.jpg"
    png = IMAGES_448 / target / f"{post_index}.png"
    if jpg.exists(): return True, str(jpg)
    if png.exists(): return True, str(png)
    return False, None

def build_all_manifests():
    for t in TARGETS:
        (MANIFESTS / t).mkdir(parents=True, exist_ok=True)
        posts    = load_json(RAW / t / "post.json")
        comments = load_json(RAW / t / "comment.json")

        for s in SPLITS:
            items = load_json(RAW / t / f"{s}.json")
            rows = []
            for it in items:
                chain = it.get("index", [])
                if not chain:
                    continue
                post_index = chain[0]
                comment_id = chain[-1] if len(chain) > 1 else None

                has_img, img_path = _img_448_exists(t, post_index)
                post_rec = posts.get(post_index, {})  # post.json keyed by int
                # comment.json is often keyed by string; try both
                com_rec  = {}
                if comment_id is not None:
                    com_rec = comments.get(str(comment_id)) or comments.get(comment_id) or {}

                rows.append({
                    # ids
                    "target": t,
                    "split": s,
                    "post_index": post_index,
                    "comment_id": comment_id,           # None if stance is for the post itself
                    "chain_ids": chain,                 # [post_index, ..., comment_id?]
                    "depth": max(0, len(chain) - 1),

                    # label
                    "stance": it.get("label"),

                    # text fields we’ll actually feed to models
                    "title": post_rec.get("title"),
                    "selftext": post_rec.get("selftext"),
                    "comment_text": com_rec.get("body"),

                    # minimal meta that can help debugging/filters later
                    "author": (com_rec.get("author") if com_rec else post_rec.get("author")),
                    "created_utc": (com_rec.get("created_utc") if com_rec else post_rec.get("created_utc")),

                    # vision hooks (we keep text-centric manifest; image optional)
                    "has_image": bool(has_img),
                    "image_path": img_path,             # 448 if exists, else None

                    # placeholders to be filled during enrichment steps
                    "captions": [],                     # list of {source, text, ts?}
                    "ocrs": []                          # list of {source, text, ts?}
                })
            outp = MANIFESTS / t / f"{s}.jsonl"
            write_jsonl(outp, rows)
            print(f"{t}/{s}: wrote {len(rows)} rows -> {outp}")

if __name__ == "__main__":
    build_all_manifests()
