# Minimal ViT wrapper used by train_vit.py / infer_vit.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

class ViTSoftPrompt(nn.Module):
    """
    Minimal wrapper around ViTForImageClassification.
    Notes:
      - `c` (prompt_tokens) is accepted for API-compat but not used here.
      - Expects inputs already resized/normalized (as in your DataLoader).
      - Returns logits of shape [B, num_labels].
    """
    def __init__(self, vit_id: str, c: int = 0, num_labels: int = 3):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            vit_id, num_labels=num_labels
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        return out.logits
