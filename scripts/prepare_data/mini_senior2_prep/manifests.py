
from pathlib import Path
import json
from typing import Dict, Iterable, Optional, List

from .config import TARGETS, PREP, MANIFESTS, IMAGES_448
from .io_utils import load_json

def _has_image_path(target: str, post_index: str) -> Optional[str]:
    pjpg = IMAGES_448/target/f"{post_index}.jpg"
    if pjpg.exists(): return str(pjpg)
    ppng = IMAGES_448/target/f"{post_index}.png"
    if ppng.exists(): return str(ppng)
    return None

def _load_ocr_cache_map(target: str) -> Dict[str, List[dict]]:
    cache_dir = PREP/"ocr"/target
    if not cache_dir.exists(): return {}
    out = {}
    for fp in sorted(cache_dir.glob("*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            pid = str(obj.get("post_index") or fp.stem)
            lines = obj.get("lines") or []
            out[pid] = lines if isinstance(lines, list) else []
        except Exception:
            pass
    return out

def _load_caption_map(target: str, captions_root: Optional[Path]) -> Dict[str, str]:
    if not captions_root: return {}
    tdir = captions_root/target
    if not tdir.exists(): return {}
    out = {}
    for fp in sorted(tdir.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                pid = str(obj.get("post_index"))
                cap = obj.get("caption")
                if pid and isinstance(cap, str) and cap.strip():
                    out[pid] = cap.strip()
            except Exception:
                continue
    return out

def _load_label_map(raw_dir: Optional[Path], target: str, split: str):
    if raw_dir is None: return {}
    norm = {"favor":"favor","support":"favor","pro":"favor",
            "against":"against","oppose":"against","con":"against",
            "none":"none","neutral":"none","other":"none", None: None}
    p = raw_dir/target/f"{split}.json"
    if not p.exists(): return {}
    rows = load_json(p)
    lab = {}
    for it in rows:
        idx = it.get("index") or []
        pid = str(idx[0]) if len(idx)>=1 else None
        cid = idx[1] if len(idx)>=2 else None
        stance = norm.get(it.get("stance"))
        if pid is not None:
            lab[(pid, cid)] = stance
    return lab

def build_unified_manifests(raw_dir: Optional[Path], captions_root: Optional[Path],
                            targets: Optional[Iterable[str]]=None,
                            splits: Iterable[str]=("train","valid","test")) -> Dict[str, Dict[str, str]]:
    targets = list(targets or TARGETS)
    written = {}
    for t in targets:
        # prefer raw mirrors if present; else rely on raw_dir
        post_map = None; comment_map = None
        mir_p = PREP/"raw_mirror"/t/"post.json"
        mir_c = PREP/"raw_mirror"/t/"comment.json"
        if mir_p.exists() and mir_c.exists():
            post_map = load_json(mir_p)
            comment_map = load_json(mir_c)
        else:
            # try raw_dir
            if raw_dir and (raw_dir/t/"post.json").exists() and (raw_dir/t/"comment.json").exists():
                post_map = load_json(raw_dir/t/"post.json")
                comment_map = load_json(raw_dir/t/"comment.json")
            else:
                raise FileNotFoundError(f"Need post.json & comment.json for {t}")

        ocr_map = _load_ocr_cache_map(t)
        cap_map = _load_caption_map(t, captions_root)

        written[t] = {}
        for s in splits:
            items = None
            mir_s = PREP/"raw_mirror"/t/f"{s}.json"
            if mir_s.exists():
                items = load_json(mir_s)
            else:
                if raw_dir and (raw_dir/t/f"{s}.json").exists():
                    items = load_json(raw_dir/t/f"{s}.json")
            if items is None:
                raise FileNotFoundError(f"Missing split {s} for {t}")

            label_map = _load_label_map(raw_dir, t, s)
            out_dir = MANIFESTS/t; out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir/f"{s}.jsonl"

            with out_path.open("w", encoding="utf-8") as fout:
                for it in items:
                    idx = it.get("index") or []
                    pid = str(idx[0]) if len(idx)>=1 else None
                    cid = idx[1] if len(idx)>=2 else None
                    if pid is None: continue

                    post = (post_map or {}).get(pid, {})
                    comm = (comment_map or {}).get(str(cid), {}) if cid is not None else {}

                    img_path = _has_image_path(t, pid)
                    ocrs = ocr_map.get(pid, [])
                    caption = cap_map.get(pid)

                    row = {
                        "target": t,
                        "split": s,
                        "post_index": pid,
                        "comment_id": cid,
                        "chain_ids": it.get("index_chain") or it.get("chain_ids") or [pid],
                        "depth": it.get("depth", 0),
                        "stance": label_map.get((pid, cid)),
                        "title": post.get("title"),
                        "selftext": post.get("selftext"),
                        "comment_text": comm.get("body") or it.get("comment_text"),
                        "author": it.get("author"),
                        "created_utc": it.get("created_utc"),
                        "image_path": img_path,
                        "has_image": bool(img_path),
                        "ocrs": ocrs if isinstance(ocrs, list) else [],
                        "caption_blip": caption,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            written[t][s] = str(out_path)
    return written
