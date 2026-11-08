"""Merge a training run into a standalone bf16 HF checkpoint (+ tokenizer) loadable by vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _run_meta(run: Path, src: Path) -> tuple[str, str]:
    """(method, base model name) from run_info.json, config.yaml or src/tinylora_config.json."""
    meta: dict = {}
    if (run / "run_info.json").exists():
        meta.update(json.loads((run / "run_info.json").read_text()))
    if (run / "config.yaml").exists():
        cfg = yaml.safe_load((run / "config.yaml").read_text()) or {}
        meta.setdefault("method", cfg.get("method"))
        meta.setdefault("model_name", cfg.get("model_name") or cfg.get("base_model"))
    if isinstance(meta.get("config"), dict) and not meta.get("model_name"):
        # t2s.train_grpo stores its whole config dict under "config".
        meta["model_name"] = meta["config"].get("model_name") or meta["config"].get("base_model")
    tl_cfg = src / "tinylora_config.json"
    if not meta.get("model_name") and tl_cfg.exists():
        meta["model_name"] = json.loads(tl_cfg.read_text()).get("base_model")
    method = meta.get("method")
    if not method:
        raise ValueError(f"cannot determine method for {run}")
    return method, meta.get("model_name") or meta.get("base_model") or DEFAULT_MODEL


def _load_merged(method: str, base: str, src: Path, device: str):
    """full: reload the saved model; lora/lora_fa/qlora: bf16 base + PEFT merge (for qlora the adapter
    trained against the NF4 base is merged into the original bf16 weights); tinylora*: bf16 base +
    TinyLoRA merge."""
    if method == "full":
        return AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    if method in ("lora", "lora_fa", "qlora"):
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device)
        return PeftModel.from_pretrained(model, src).merge_and_unload()
    if method.startswith("tinylora"):
        from .tinylora import load_tinylora, merge_and_unload

        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device)
        load_tinylora(model, src)
        return merge_and_unload(model)
    raise ValueError(f"unknown method {method!r}")


def merge(run: Path, out: Path, adapter_dir: Path | None = None, device: str = "cuda") -> Path:
    """Merge run/final (or `adapter_dir`, e.g. a checkpoint-N dir) into a checkpoint at `out`."""
    src = adapter_dir or (run / "final")
    if not src.exists():
        raise FileNotFoundError(src)
    method, base = _run_meta(run, src)
    if (src / "tinylora_config.json").exists() and not method.startswith("tinylora"):
        method = "tinylora"
    print(f"merge: method={method} base={base} src={src} -> {out}", flush=True)

    model = _load_merged(method, base, src, device).to(torch.bfloat16)
    model.config.use_cache = True
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok_src = src if (src / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(out)
    merge_info = {"run": str(run), "method": method, "base_model": base, "adapter": str(src)}
    (out / "merge_info.json").write_text(json.dumps(merge_info, indent=2))
    print(f"saved {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="runs/<name>")
    ap.add_argument("--out", required=True, help="checkpoints/<name>")
    ap.add_argument("--adapter-dir", default=None, help="adapter dir to merge instead of <run>/final")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    merge(Path(args.run), Path(args.out), Path(args.adapter_dir) if args.adapter_dir else None, args.device)


if __name__ == "__main__":
    main()
