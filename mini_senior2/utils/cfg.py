from pathlib import Path
import yaml

def load_data_cfg(path: str|Path):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
