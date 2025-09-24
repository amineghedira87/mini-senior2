"""Text-only stance pass with LLaMA; reads prepared manifests and writes labels/scores/metrics."""
from __future__ import annotations
import argparse, json, numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from mini_senior2.models.text_llm import load_text_model, batch_generate, parse_label
from mini_senior2.utils.config_loader import load_paths
import os, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings(
    'ignore',
    message='`do_sample` is set to `False`.*temperature',
    category=UserWarning,
)

LABEL2ID = {"FAVOR":0, "AGAINST":1, "NONE":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def split_paths(base: str, target: str):
    base = Path(base)
    return base/f"{target}/train.jsonl", base/f"{target}/valid.jsonl", base/f"{target}/test.jsonl"

def _trim(s: Optional[str], lim: int) -> str:
    s = (s or "").strip()
    return s[:lim] if len(s) > lim else s

def build_index(all_rows: List[dict]) -> Dict[str, dict]:
    idx = {}
    for r in all_rows:
        chain = r.get("chain_ids") or []
        root = str(chain[0]) if chain else "?"
        cid  = r.get("comment_id") or "ROOT"
        idx[f"{root}::{cid}"] = r
    return idx

def prev_comments_text(row: dict, idx: Dict[str, dict], max_prev: int = 2) -> List[str]:
    chain = row.get("chain_ids") or []
    if row.get("depth", 0) <= 0 or len(chain) <= 2:
        return []
    root = str(chain[0]); prev_ids = chain[1:-1]
    out = []
    for cmt_id in prev_ids[-max_prev:]:
        r_prev = idx.get(f"{root}::{cmt_id}")
        if r_prev:
            txt = (r_prev.get("comment_text") or "").strip()
            if txt: out.append(txt)
    return out

def build_prompt(row: dict, target: str, idx: Dict[str, dict], max_prev: int = 2) -> str:
    # keep the same plain-instruction prompt your code-files.txt used
    title   = _trim(row.get("title"),     200)
    selftxt = _trim(row.get("selfText"),  800)
    caption = _trim(row.get("caption"),   240)
    cmt     = _trim(row.get("comment_text"), 360)
    prevs   = prev_comments_text(row, idx, max_prev=max_prev)
    image_block = "" if not caption else f"Image Caption (textual proxy): {caption}"

    instruction = (
        "You are a stance classifier.\n"
        f"Target: {target}\n\n"
        "Given Reddit content (text only), classify the stance TOWARD THE TARGET as exactly one of: FAVOR, AGAINST, or NONE.\n"
        "Respond with ONLY one word: FAVOR, AGAINST, or NONE."
    )
    parts = [instruction]
    if title:   parts.append(f"Post title: {title}")
    if selftxt: parts.append(f"Post text: {selftxt}")
    if image_block: parts.append(image_block)
    for i, p in enumerate(prevs[-2:], 1):
        parts.append(f"Previous comment {i}: {p}")
    if cmt: parts.append(f"Current comment: {cmt}")
    parts.append("Answer:")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["bitcoin","tesla","post"])
    ap.add_argument("--split", required=True, choices=["train","valid","test"])
    ap.add_argument("--manifests_root", default="/kaggle/working/mini-senior2_data/prepared/manifests")
    ap.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--max_input_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--out_dir", default="/kaggle/working")
    ap.add_argument("--config", default="mini_senior2/configs/paths.yaml")
    ap.add_argument("--hf_token", default="")
    args = ap.parse_args()

    paths = load_paths(args.config)
    if args.manifests_root == "/kaggle/working/mini-senior2_data/prepared/manifests":
        args.manifests_root = paths["manifests_root"]
    if args.out_dir == "/kaggle/working":
        args.out_dir = paths["outputs_dir"]
    hf_token = args.hf_token or paths["hf_token"]

    tr_p, va_p, te_p = split_paths(args.manifests_root, args.target)
    train_rows, valid_rows, test_rows = [read_jsonl(str(p)) for p in (tr_p, va_p, te_p)]
    rows = {"train":train_rows, "valid":valid_rows, "test":test_rows}[args.split]
    idx  = build_index(train_rows + valid_rows + test_rows)

    prompts = [build_prompt(r, args.target, idx, max_prev=2) for r in tqdm(rows, desc="prompts", leave=True)]
    tok, mdl, mode = load_text_model(args.model_id, max_input_tokens=args.max_input_tokens, hf_token=hf_token)
    raw = batch_generate(tok, mdl, prompts, max_input_tokens=args.max_input_tokens,
                         max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    preds = [parse_label(x) for x in raw]

    lab2id = {"FAVOR":0,"AGAINST":1,"NONE":2}
    y_true = np.array([lab2id[r["stance"]] for r in rows], dtype=np.int64)
    y_pred = np.array([lab2id[p] for p in preds], dtype=np.int64)

    def f1c(c):
        tp = int(((y_pred==c) & (y_true==c)).sum())
        fp = int(((y_pred==c) & (y_true!=c)).sum())
        fn = int(((y_pred!=c) & (y_true==c)).sum())
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        return (2*p*r/(p+r)) if (p+r)>0 else 0.0

    f1_favor   = f1c(0)
    f1_against = f1c(1)
    f1_none    = f1c(2)
    macro2 = (f1_favor + f1_against) / 2.0
    macro3 = (f1_favor + f1_against + f1_none) / 3.0

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = np.zeros((len(preds), 3), dtype=np.float32)
    for i,p in enumerate(preds):
        scores[i, lab2id[p]] = 1.0
    np.save(out / f"llama_{args.split}_scores.npy", scores)
    (out / f"llama_{args.target}_{args.split}_labels.json").write_text(
        json.dumps({"target": args.target, "split": args.split, "labels": preds}, indent=2), encoding="utf-8"
    )
    (out / f"llama_{args.target}_{args.split}_metrics.txt").write_text(
        f"F1-avg(F/A)={macro2:.4f}  Macro-F1(3)={macro3:.4f}\n"
        f"F1-FAVOR={f1_favor:.4f}\n"
        f"F1-AGAINST={f1_against:.4f}\n",
        encoding="utf-8"
    )

    print(f"{args.target}/{args.split}  F1-avg(F/A)={macro2:.4f}  Macro-F1(3)={macro3:.4f}")
    print(f"F1-FAVOR={f1_favor:.4f}  F1-AGAINST={f1_against:.4f}")
    print("💾 saved: llama_{split}_scores.npy, labels.json, metrics.txt".format(split=args.split))

if __name__ == "__main__":
    main()
