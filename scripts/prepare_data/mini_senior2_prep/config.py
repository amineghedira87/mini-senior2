
from pathlib import Path

# where our prepared tree lives in this session
PREP = Path("/kaggle/working/mini-senior2_data")

# standard subpaths we use
MANIFESTS = PREP / "prepared" / "manifests"
IMAGES_448 = PREP / "prepared" / "images" / "448"

# targets in this dataset
TARGETS = ["bitcoin", "tesla", "post"]
