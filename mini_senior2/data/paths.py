from pathlib import Path
import subprocess

def ensure_prepared_from_kaggle(slug: str, work_root: str) -> Path:
    ds = Path(f"/kaggle/input/{slug}")
    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)

    # If already extracted, just return
    if (work/"manifests").exists() and (work/"images"/"448").exists():
        return work

    # 1) unzip any zips inside dataset into work_root
    zips = list(ds.rglob("*.zip"))
    for z in zips:
        subprocess.run(["unzip","-o","-q",str(z),"-d",str(work)], check=False)

    # 2) also copy if uploaded as folders
    candidates = [ds/"mini-senior2_data"/"prepared", ds/"prepared"]
    for c in candidates:
        if c.exists() and c.is_dir():
            subprocess.run(["cp","-r",str(c)+"/.",str(work)], check=False)

    return work
