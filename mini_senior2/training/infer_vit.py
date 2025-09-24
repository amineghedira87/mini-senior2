"""Infer ViT on TEST; save logits for image rows (M x 3) and a boolean mask (N)."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from mini_senior2.models.vit_prompt import ViTSoftPrompt
from mini_senior2.utils.config_loader import load_paths

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
    ap.add_argument("--ckpt", default="/kaggle/working/vit_prompt_best.pt")
    ap.add_argument("--out_dir", default="/kaggle/working")
    ap.add_argument("--batch_eval", type=int, default=96)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--config", default="mini_senior2/configs/paths.yaml")
    args = ap.parse_args()

    paths = load_paths(args.config)
    if args.manifests_root == "/kaggle/working/mini-senior2_data/prepared/manifests":
        args.manifests_root = paths["manifests_root"]
    if args.images_root == "/kaggle/working/mini-senior2_data/prepared/images/448":
        args.images_root = paths["images_root"]
    if args.out_dir == "/kaggle/working":
        args.out_dir = paths["outputs_dir"]

    base = Path(args.manifests_root) / args.target
    test_rows = [json.loads(l) for l in open(base/"test.jsonl","r",encoding="utf-8")]
    mask = np.array([has_image(r) for r in test_rows], dtype=bool)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    class TestSet(Dataset):
        def __init__(self, rows): self.rows = [r for r in rows if has_image(r)]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            p = resolve_image_path(r, args.images_root)
            try: img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (args.img_size, args.img_size), (0,0,0))
            return tfm(img)

    ds = TestSet(test_rows)
    loader = DataLoader(ds, batch_size=args.batch_eval, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTSoftPrompt(args.vit_id, c=args.prompt_tokens, num_labels=3)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    logits_all = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            logits_all.append(model(xb).detach().cpu().numpy())
    vit_logits = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0,3), dtype=np.float32)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out/"vit_test_logits.npy", vit_logits)  # shape (M,3)
    np.save(out/"vit_test_mask.npy", mask)          # shape (N,)
    print(f"💾 saved: {out/'vit_test_logits.npy'} , {out/'vit_test_mask.npy'} (shape={vit_logits.shape}, mask_true={mask.sum()})")

if __name__ == "__main__":
    main()
