from pathlib import Path

DATA_ROOT = Path(r"/kaggle/input/mini-senior2-data/mini-senior2_data")
RAW = DATA_ROOT / "raw"
PREP = Path("/kaggle/working/mini-senior2_data") / "prepared"

TARGETS = ["bitcoin", "tesla", "post"]
SPLITS  = ["train", "valid", "test"]

IMAGES_ORIG = PREP / "images" / "orig"
IMAGES_448  = PREP / "images" / "448"

REPORTS_DIR = PREP / "reports"
MANIFESTS   = PREP / "manifests"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT_SEC = 12
MAX_WORKERS = 16
RETRIES     = 2
