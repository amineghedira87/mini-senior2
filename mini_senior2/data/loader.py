
from pathlib import Path
import json
from typing import List, Optional, Dict

from transformers import AutoTokenizer
from torch.utils.data import Dataset

LABEL_MAP = {"favor":0, "against":1, "none":2}

def join_nonempty(parts: List[Optional[str]], sep=" \n"):
    return sep.join([p.strip() for p in parts if isinstance(p, str) and p.strip()])

def make_text(obj: Dict, include_ocr=True, include_caption=True) -> Optional[str]:
    ocr_text = None
    if include_ocr and isinstance(obj.get("ocrs"), list) and obj["ocrs"]:
        ocr_text = " ".join([seg.get("text","") for seg in obj["ocrs"] if isinstance(seg, dict)])
    cap = obj.get("caption_blip") if include_caption else None
    return join_nonempty([
        obj.get("title"),
        obj.get("selftext"),
        obj.get("comment_text"),
        f"[OCR] {ocr_text}" if ocr_text else None,
        f"[CAPTION] {cap}" if cap else None
    ])

class StanceJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer_name="roberta-base",
                 max_length=512, multimodal_only=False, use_ocr=True, use_caption=True, drop_unlabeled=True):
        self.rows = []
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                lab = obj.get("stance")
                if drop_unlabeled and lab not in LABEL_MAP:
                    continue
                if multimodal_only and not (obj.get("image_path") or obj.get("ocrs") or obj.get("caption_blip")):
                    continue
                text = make_text(obj, include_ocr=use_ocr, include_caption=use_caption)
                if not text:
                    continue
                self.rows.append({"text": text, "label": LABEL_MAP.get(lab)})

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k:v.squeeze(0) for k,v in enc.items()}
        if r["label"] is not None:
            item["labels"] = r["label"]
        return item
