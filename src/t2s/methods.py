"""Model + tokenizer construction for the five fine-tuning methods (full, lora, lora_fa, qlora, tinylora)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

METHODS = ("full", "lora", "lora_fa", "qlora", "tinylora")
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
GC_KWARGS = {"use_reentrant": False}


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _lora_config(cfg: dict):
    from peft import LoraConfig

    lc = cfg.get("lora", {}) or {}
    return LoraConfig(
        r=int(lc.get("r", 64)),
        lora_alpha=int(lc.get("alpha", 128)),
        lora_dropout=float(lc.get("dropout", 0.05)),
        target_modules=list(lc.get("targets", ALL_LINEAR)),
        bias="none",
        task_type="CAUSAL_LM",
    )


def _count_params(model) -> tuple[int, int]:
    """(trainable, total), each unique parameter counted once."""
    trainable = total = 0
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        if p.__class__.__name__ == "Params4bit":
            n *= 2  # bnb packs two 4-bit values per stored element
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _build_qlora(name: str, cfg: dict, attn: str, gc: bool, device: str | torch.device):
    from peft import get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map={"": torch.device(device).index or 0},
    )
    base_params = _count_params(model)[1]
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gc, gradient_checkpointing_kwargs=GC_KWARGS
    )
    return get_peft_model(model, _lora_config(cfg)), base_params


def _freeze_lora_A(model) -> None:
    """LoRA-FA: A stays at its (kaiming-uniform) init and only the zero-init B trains."""
    n_frozen = 0
    for n, p in model.named_parameters():
        if "lora_A" in n:
            p.requires_grad_(False)
            n_frozen += 1
    assert n_frozen > 0


def _apply_tinylora(model, cfg: dict) -> None:
    from .tinylora import apply_tinylora

    tc = cfg.get("tinylora", {}) or {}
    apply_tinylora(
        model,
        u=int(tc.get("u", 13)),
        r=int(tc.get("r", 2)),
        tie=str(tc.get("tie", "all")),
        targets=tuple(tc.get("targets", ALL_LINEAR)),
        seed=int(tc.get("seed", 0)),
        share_P=bool(tc.get("share_P", False)),
    )


def build_model(method: str, cfg: dict, device: str | torch.device = "cuda"):
    """Return (model, tokenizer, info) for `method`, configured from the parsed YAML `cfg`."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    name = cfg.get("model_name", DEFAULT_MODEL)
    attn = cfg.get("attn_implementation", "sdpa")
    gc = bool(cfg.get("gradient_checkpointing", True))
    tok = load_tokenizer(name)

    if method == "qlora":
        model, base_params = _build_qlora(name, cfg, attn, gc, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation=attn).to(
            device
        )
        base_params = _count_params(model)[1]
        if gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
        if method == "full":
            for p in model.parameters():
                p.requires_grad_(True)
        elif method in ("lora", "lora_fa"):
            from peft import get_peft_model

            model = get_peft_model(model, _lora_config(cfg))
            if method == "lora_fa":
                _freeze_lora_A(model)
        else:
            _apply_tinylora(model, cfg)

    model.config.use_cache = False
    trainable, total = _count_params(model)
    info = {
        "method": method,
        "model_name": name,
        "trainable_params": trainable,
        "total_params": total,
        "base_params": base_params,
        "pct": 100.0 * trainable / total,
        "pct_of_base": 100.0 * trainable / base_params,
    }
    return model, tok, info
