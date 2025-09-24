"""
LLaMA text stance helpers:
- load_text_model: loads tokenizer + model (tries 4-bit if available / requested), left padding.
- batch_generate: batched greedy generation with a tqdm progress bar (as in your code-flow).
- parse_label: simple contains-based mapping to FAVOR / AGAINST / NONE (unchanged).
"""
from __future__ import annotations
from typing import List, Tuple
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

def _have_bnb() -> bool:
    try:
        import importlib.metadata, bitsandbytes  # noqa: F401
        importlib.metadata.version("bitsandbytes")
        return True
    except Exception:
        return False

def load_text_model(model_id: str, max_input_tokens: int = 2048, hf_token: str = "") -> Tuple[object, object, str]:
    """
    Matches your original behavior:
      - If bitsandbytes present and USE_BNB!=0, try 4-bit, else fp16/fp32.
      - Always set tokenizer.pad_token from eos if missing.
      - Always set tokenizer.padding_side='left' (decoder-only best practice).
    """
    token = hf_token or None
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    use_4bit = False
    if os.environ.get("USE_BNB", "0") not in ("0", "false", "False"):
        if _have_bnb():
            try:
                compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                qcfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                    quantization_config=qcfg,
                    token=token,
                )
                use_4bit = True
            except Exception as e:
                print(f"[WARN] 4-bit path failed ({e}); falling back to fp16/fp32.")

    if not use_4bit:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            token=token,
        )
    return tok, mdl, ("4bit" if use_4bit else ("fp16" if torch.cuda.is_available() else "fp32"))

def batch_generate(tok, mdl, prompts: List[str], *, max_input_tokens: int = 2048, max_new_tokens: int = 6, batch_size: int = 4) -> List[str]:
    """
    EXACT batch loop shape from your flow: tqdm over batches, greedy, slice off prompt by in_len.
    """
    outs: List[str] = []
    total = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", total=total, leave=True):
        batch = prompts[i:i+batch_size]
        toks = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )
        in_len = toks["input_ids"].shape[1]
        for j in range(len(batch)):
            outs.append(tok.decode(gen[j][in_len:], skip_special_tokens=True).strip())
    return outs

def parse_label(text: str) -> str:
    if not text: return "NONE"
    t = text.strip().lower()
    if "favor" in t: return "FAVOR"
    if "against" in t: return "AGAINST"
    if "none" in t or "neutral" in t: return "NONE"
    return "NONE"
