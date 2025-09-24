from pathlib import Path
import json
from typing import Dict, Tuple, Optional

NORMALIZE = {
    "favor":"favor", "support":"favor", "pro":"favor",
    "against":"against", "oppose":"against", "con":"against",
    "none":"none", "neutral":"none", "other":"none", None: None
}

Key = Tuple[str, Optional[str]]  # (post_index, comment_id)

def load_label_map(raw_dir: Path, target: str, split: str) -> Dict[Key, str]:
    p = raw_dir / target / f"{split}.json"
    with p.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    m: Dict[Key, str] = {}
    for it in rows:
        idx = it.get("index") or []
        post_index = str(idx[0]) if len(idx) >= 1 else None
        comment_id = idx[1] if len(idx) >= 2 else None
        stance = NORMALIZE.get(it.get("stance"))
        if post_index is None or stance is None:
            continue
        m[(post_index, comment_id)] = stance
    return m

def merge_split(manifest_in: Path, manifest_out: Path, label_map: Dict[Key, str]) -> Dict[str,int]:
    n = n_labeled = 0
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_in.open("r", encoding="utf-8") as fin, manifest_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            pid = str(obj.get("post_index"))
            cid = obj.get("comment_id")  # may be None
            obj["stance"] = label_map.get((pid, cid))
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            if obj["stance"] is not None:
                n_labeled += 1
    return {"rows": n, "labeled": n_labeled}

def merge_all(raw_dir: Path, prepared_manifests: Path, target: str):
    out = {}
    for split in ["train","valid","test"]:
        lm = load_label_map(raw_dir, target, split)
        src = prepared_manifests/target/f"{split}.with_captions.jsonl"
        dst = prepared_manifests/target/f"{split}.labeled.jsonl"
        stats = merge_split(src, dst, lm)
        out[split] = {"in": str(src), "out": str(dst), **stats}
    return out
