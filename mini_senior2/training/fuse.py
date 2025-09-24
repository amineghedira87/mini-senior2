"""Late fusion on TEST: combine text scores + ViT logits with mask alignment.
Restores original code-flow math:
 - text: safe_log_softmax (so one-hot doesn’t dominate)
 - vit:  softmax -> place under mask -> safe_log_softmax
 - fuse in log-prob space
 - NONE gate: if predicted NONE but P_none < tau, re-pick from FAVOR/AGAINST
Saves metrics (avg F1 over F/A, Macro-F1(3), F1 per class) and a CSV.
"""
from __future__ import annotations
import argparse, json, numpy as np
from pathlib import Path
from mini_senior2.utils.config_loader import load_paths

LABEL2ID = {"FAVOR":0,"AGAINST":1,"NONE":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def read_jsonl(path: str):
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def safe_log_softmax(x, axis=1):
    # Works whether x are logits, one-hot, or generic scores
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    p  = ex / np.sum(ex, axis=axis, keepdims=True)
    p  = np.clip(p, 1e-8, 1.0)
    return np.log(p)

def prf(y_true, y_pred, cls_id):
    tp = int(((y_pred==cls_id) & (y_true==cls_id)).sum())
    fp = int(((y_pred==cls_id) & (y_true!=cls_id)).sum())
    fn = int(((y_pred!=cls_id) & (y_true==cls_id)).sum())
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = (2*p*r/(p+r)) if (p+r)>0 else 0.0
    return p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["bitcoin","tesla","post"])
    ap.add_argument("--manifests_root", default="/kaggle/working/mini-senior2_data/prepared/manifests")
    ap.add_argument("--text_scores", default="/kaggle/working/llama_test_scores.npy")  # [N,3]
    ap.add_argument("--vit_logits",  default="/kaggle/working/vit_test_logits.npy")   # [M,3]
    ap.add_argument("--mask",        default="/kaggle/working/vit_test_mask.npy")     # [N]
    ap.add_argument("--w_text", type=float, default=0.60)
    ap.add_argument("--tau",    type=float, default=0.60)
    ap.add_argument("--out_dir", default="/kaggle/working")
    ap.add_argument("--config", default="mini_senior2/configs/paths.yaml")
    args = ap.parse_args()

    paths = load_paths(args.config)
    if args.manifests_root == "/kaggle/working/mini-senior2_data/prepared/manifests":
        args.manifests_root = paths["manifests_root"]
    if args.out_dir == "/kaggle/working":
        args.out_dir = paths["outputs_dir"]

    # Ground truth from manifests
    test_rows = read_jsonl(str(Path(args.manifests_root)/f"{args.target}/test.jsonl"))
    y_true = np.array([LABEL2ID[r["stance"]] for r in test_rows], dtype=np.int64)

    # Load artifacts
    text_scores = np.load(args.text_scores).astype(np.float32)  # [N,3] (often one-hot)
    vit_logits  = np.load(args.vit_logits).astype(np.float32)   # [M,3]
    mask        = np.load(args.mask).astype(bool)               # [N]

    N = text_scores.shape[0]
    M = vit_logits.shape[0]
    assert mask.shape[0] == N, f"mask length {mask.shape[0]} != N {N}"
    assert M == int(mask.sum()), f"ViT logits M={M} must equal mask sum={int(mask.sum())}"

    # ViT probabilities only for image rows
    vit_probs = softmax_np(vit_logits, axis=1)                  # [M,3]
    vit_full  = np.zeros_like(text_scores, dtype=np.float32)    # [N,3]
    vit_full[mask] = vit_probs

    # Convert both streams to log-probabilities robustly
    text_logp = safe_log_softmax(text_scores, axis=1)           # [N,3]
    vit_logp  = safe_log_softmax(vit_full,  axis=1)             # [N,3]

    # Fuse in log-prob space
    w = float(args.w_text)
    mix_logp = w*text_logp + (1.0 - w)*vit_logp

    # None gate
    mix_p = np.exp(mix_logp - np.max(mix_logp, axis=1, keepdims=True))
    mix_p = mix_p / np.sum(mix_p, axis=1, keepdims=True)
    y_pred = np.argmax(mix_logp, axis=1).astype(np.int64)

    NONE, FAVOR, AGAINST = LABEL2ID["NONE"], LABEL2ID["FAVOR"], LABEL2ID["AGAINST"]
    tau = float(args.tau)
    idx_none_weak = (y_pred == NONE) & (mix_p[:, NONE] < tau)
    if np.any(idx_none_weak):
        fa = np.argmax(mix_logp[idx_none_weak][:, [FAVOR, AGAINST]], axis=1)
        # map 0->FAVOR, 1->AGAINST
        y_pred[idx_none_weak] = np.where(fa == 0, FAVOR, AGAINST)

    # Metrics
    fav = prf(y_true, y_pred, FAVOR)
    agn = prf(y_true, y_pred, AGAINST)
    non = prf(y_true, y_pred, NONE)
    fa_avg = (fav[2] + agn[2]) / 2.0
    macro3 = (fav[2] + agn[2] + non[2]) / 3.0

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"fusion_{args.target}_test_metrics.txt").write_text(
        f"F1-avg(F/A)={fa_avg:.4f}  Macro-F1(3)={macro3:.4f}\n"
        f"F1-FAVOR={fav[2]:.4f}\n"
        f"F1-AGAINST={agn[2]:.4f}\n",
        encoding="utf-8"
    )

    # Also save per-row CSV (useful for audits)
    import csv
    csv_p = out / f"fusion_{args.target}_test.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["idx","true","pred","pred_label",
                       "mix_logp_favor","mix_logp_against","mix_logp_none"])
        for i,(yt,yp,vec) in enumerate(zip(y_true, y_pred, mix_logp)):
            wcsv.writerow([i, int(yt), int(yp), ID2LABEL[int(yp)],
                           float(vec[0]), float(vec[1]), float(vec[2])])

    print(f"{args.target}/test  Fusion F1-avg(F/A)={fa_avg:.4f}  Macro-F1(3)={macro3:.4f}")
    print(f"F1-FAVOR={fav[2]:.4f}  F1-AGAINST={agn[2]:.4f}")
    print(f"💾 saved: {csv_p.name} and metrics file")

if __name__ == "__main__":
    main()
