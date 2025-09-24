"""Train ViT soft prompts on images; early-stop on VALID; save best checkpoint."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from mini_senior2.models.vit_prompt import ViTSoftPrompt
from mini_senior2.utils.config_loader import load_paths

LABEL2ID = {"FAVOR":0, "AGAINST":1, "NONE":2}

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def has_image(r: dict) -> bool:
    return bool(r.get("has_image") and r.get("image_path"))

def resolve_image_path(r: dict, images_root: str) -> str:
    p = (r.get("image_path") or "").strip()
    if not p: return ""
    from os.path import isabs, exists, join
    if isabs(p) and exists(p): return p
    cand = join(images_root, p.lstrip("/"))
    return cand if exists(cand) else p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["bitcoin","tesla","post"])
    ap.add_argument("--manifests_root", default="/kaggle/working/mini-senior2_data/prepared/manifests")
    ap.add_argument("--images_root",    default="/kaggle/working/mini-senior2_data/prepared/images/448")
    ap.add_argument("--vit_id", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--prompt_tokens", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_train", type=int, default=32)
    ap.add_argument("--batch_eval",  type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--out_ckpt", default="/kaggle/working/vit_prompt_best.pt")
    ap.add_argument("--config", default="mini_senior2/configs/paths.yaml")
    args = ap.parse_args()

    paths = load_paths(args.config)
    if args.manifests_root == "/kaggle/working/mini-senior2_data/prepared/manifests":
        args.manifests_root = paths["manifests_root"]
    if args.images_root == "/kaggle/working/mini-senior2_data/prepared/images/448":
        args.images_root = paths["images_root"]

    base = Path(args.manifests_root) / args.target
    train_rows = read_jsonl(str(base/"train.jsonl"))
    valid_rows = read_jsonl(str(base/"valid.jsonl"))

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    class Set(Dataset):
        def __init__(self, rows): self.rows = [r for r in rows if has_image(r)]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            y = LABEL2ID[r["stance"]]
            p = resolve_image_path(r, args.images_root)
            try: img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (args.img_size, args.img_size), (0,0,0))
            return tfm(img), torch.tensor(y, dtype=torch.long)

    train_ds, valid_ds = Set(train_rows), Set(valid_rows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_train, shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_eval,  shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTSoftPrompt(args.vit_id, c=args.prompt_tokens, num_labels=3).to(device)

    counts = Counter()
    for _, y in DataLoader(train_ds, batch_size=512, shuffle=False):
        counts.update(y.tolist())
    tot = sum(counts.values()) + 1e-9
    weights = torch.tensor(
        [tot/max(1,counts.get(0,0)), tot/max(1,counts.get(1,0)), tot/max(1,counts.get(2,0))],
        dtype=torch.float32, device=device
    )
    weights = weights / weights.mean()

    crit = nn.CrossEntropyLoss(weight=weights)
    opt  = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    best_val, patience = float("inf"), 0
    for ep in range(1, args.epochs+1):
        model.train()
        tot_loss, ok, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            tot_loss += float(loss.item()) * xb.size(0)
            ok += (logits.argmax(1) == yb).sum().item(); n += xb.size(0)
        tr_loss, tr_acc = tot_loss/max(1,n), ok/max(1,n)

        model.eval()
        tot_loss, ok, n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = crit(logits, yb)
                tot_loss += float(loss.item()) * xb.size(0)
                ok += (logits.argmax(1) == yb).sum().item(); n += xb.size(0)
        va_loss, va_acc = tot_loss/max(1,n), ok/max(1,n)
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr_loss:.3f} acc={tr_acc*100:.1f}% | valid_loss={va_loss:.3f} acc={va_acc*100:.1f}%")

        if va_loss < best_val - 1e-4:
            best_val, patience = va_loss, 0
            torch.save(model.state_dict(), args.out_ckpt)
            print(f"✅ Saved best to {args.out_ckpt}")
        else:
            patience += 1
            if patience >= args.patience:
                print("⏹️ Early stop"); break

if __name__ == "__main__":
    main()
