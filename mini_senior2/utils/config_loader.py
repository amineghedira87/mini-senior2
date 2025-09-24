import os, yaml
from pathlib import Path

def load_paths(cfg_path: str):
    p = Path(cfg_path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    # env override if provided
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf:
        data["hf_token"] = hf
    return data
