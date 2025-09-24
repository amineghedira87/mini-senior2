# mini-senior2
Multimodal (image+text) multi-turn stance detection.  
This repo hosts training code and data-prep scripts.

## Layout
- `scripts/prepare_data/mini_senior2_prep/` – all data prep utilities (download, resize, OCR, captions, manifests).
- `mini_senior2/` – training package (data loaders, models, training loop, configs).

## Dataset
Prepared dataset is published on Kaggle (to be added).  
Manifests: `*.with_captions.jsonl`; images: `images/orig` and `images/448`.

