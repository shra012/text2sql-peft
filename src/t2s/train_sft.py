"""Completion-only SFT for all five methods; writes config.yaml, train_log.csv, run_info.json, final/."""

from __future__ import annotations

import os

# Must be set before torch initialises CUDA / tokenizers spawn threads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .methods import build_model

IM_END = "<|im_end|>"
LOG_COLUMNS = [
    "step",
    "loss",
    "lr",
    "tokens_per_s",
    "step_time_s",
    "tokens",
    "completion_tokens",
    "grad_norm",
    "peak_alloc_gb",
    "device_used_gb",
]


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_example(tok, messages: list[dict], completion: str) -> tuple[list[int], list[int]]:
    """input_ids = chat prompt + completion + <|im_end|> + "\\n"; labels only on completion + <|im_end|>."""
    prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(prompt_ids, dict):  # some transformers versions return a BatchEncoding
        prompt_ids = prompt_ids["input_ids"]
    comp_ids = tok(completion + IM_END, add_special_tokens=False)["input_ids"]
    nl_ids = tok("\n", add_special_tokens=False)["input_ids"]
    input_ids = list(prompt_ids) + comp_ids + nl_ids
    labels = [-100] * len(prompt_ids) + comp_ids + [-100] * len(nl_ids)
    return input_ids, labels


def select_rows(
    rows: list[dict],
    train_limit: int | None,
    max_prompt_tokens: int | None,
    drop_fit_levels=("truncated",),
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Method- and tokenizer-independent training subset.

    Drops rows whose fit_level is in drop_fit_levels (the schema had to be truncated, so the gold
    SQL may reference cut tables) or whose prompt_tokens exceed max_prompt_tokens, shuffles with
    `seed` and keeps the first train_limit rows.
    """
    stats = {"n_rows": len(rows)}
    kept = [r for r in rows if r.get("fit_level") not in drop_fit_levels]
    stats["dropped_fit_level"] = len(rows) - len(kept)
    if max_prompt_tokens:
        n_before = len(kept)
        kept = [r for r in kept if int(r.get("prompt_tokens", 0)) <= max_prompt_tokens]
        stats["dropped_prompt_tokens"] = n_before - len(kept)
    stats["n_eligible"] = len(kept)
    random.Random(seed).shuffle(kept)
    if train_limit:
        kept = kept[:train_limit]
    stats["n_selected"] = len(kept)
    return kept, stats


def build_examples(rows: list[dict], tok, max_len: int) -> tuple[list[dict], int]:
    """Tokenize rows, skipping those longer than max_len. Returns (examples, n_dropped_too_long)."""
    examples, dropped = [], 0
    for r in rows:
        ids, labels = tokenize_example(tok, r["messages"], r["completion"])
        if len(ids) > max_len:
            dropped += 1
            continue
        examples.append({"input_ids": ids, "labels": labels})
    return examples, dropped


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor | None]:
    """Right-pad; labels are -100 on prompt and padding. attention_mask is None for a single row."""
    T = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": None if len(batch) == 1 else att}


def completion_loss_sum(model, input_ids, labels, attention_mask=None) -> tuple[torch.Tensor, int]:
    """(sum of CE over completion tokens, n completion tokens).

    The LM head is applied only to completion positions, so full-vocab logits are never
    materialised for the (much longer) prompt.
    """
    # PeftModel -> underlying *ForCausalLM (LoRA layers are injected in place).
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    logits = lm.lm_head(hidden[:, :-1][mask]).float()
    loss = F.cross_entropy(logits, tgt[mask], reduction="sum")
    return loss, int(mask.sum())


def build_optimizer(method: str, model, tcfg: dict) -> tuple[torch.optim.Optimizer, str]:
    params = [p for p in model.parameters() if p.requires_grad]
    wd = float(tcfg.get("weight_decay", 0.0))
    betas = tuple(tcfg.get("betas", (0.9, 0.999)))
    name = tcfg.get("optimizer") or ("paged_adamw_8bit" if method == "full" else "adamw")
    lr = float(tcfg["lr"])
    if name == "paged_adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=lr, betas=betas, weight_decay=wd), name
    if name == "adamw":
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd, fused=fused), name
    raise ValueError(f"unknown optimizer {name!r}")


def cosine_with_warmup(opt, warmup: int, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to 0. Unlike HF's version the first optimizer step
    already uses a non-zero lr (step s uses factor (s+1)/warmup during warmup)."""

    def factor(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup + 1) / max(1, total - warmup + 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def save_final(method: str, model, tok, out: Path) -> None:
    """TinyLoRA safetensors for tinylora, a PEFT adapter for lora/lora_fa/qlora, bf16 model for full."""
    out.mkdir(parents=True, exist_ok=True)
    if method == "tinylora":
        from .tinylora import save_tinylora

        save_tinylora(model, out)
    else:
        model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/<method>.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (schedule is compressed)")
    ap.add_argument("--out", default=None, help="run dir (default runs/<run_name>)")
    ap.add_argument("--grad-accum", type=int, default=None, help="override train.grad_accum")
    ap.add_argument("--train-file", default=None, help="override train_file")
    ap.add_argument(
        "--max-samples", type=int, default=None, help="debug: truncate the (already selected) training order"
    )
    ap.add_argument("--lr", type=float, default=None, help="override train.lr (LR sweeps)")
    ap.add_argument(
        "--longest-first",
        action="store_true",
        help="(smoke/memory tests) train on the longest examples first",
    )
    ap.add_argument("--no-save", action="store_true", help="skip checkpoint and final/ saving")
    return ap.parse_args(argv)


def _train_config(cfg: dict, args: argparse.Namespace) -> dict:
    tcfg = dict(cfg.get("train", {}) or {})
    if args.grad_accum:
        tcfg["grad_accum"] = args.grad_accum
    if args.lr is not None:
        tcfg["lr"] = args.lr
    return tcfg


def _prepare_examples(cfg: dict, tcfg: dict, tok, train_file: str, max_len: int, seed: int):
    """Select, tokenize and length-filter the training rows. Returns (examples, data stats)."""
    drop_levels = tuple(tcfg.get("drop_fit_levels", ["truncated"]) or ())
    train_limit = tcfg.get("train_limit") or cfg.get("train_limit")
    max_prompt_tokens = tcfg.get("max_prompt_tokens") or cfg.get("max_prompt_tokens")
    rows, stats = select_rows(
        load_jsonl(train_file),
        int(train_limit) if train_limit else None,
        int(max_prompt_tokens) if max_prompt_tokens else None,
        drop_levels,
        seed,
    )
    examples, stats["dropped_too_long"] = build_examples(rows, tok, max_len)
    stats["n_examples"] = len(examples)
    stats["total_tokens"] = sum(len(e["input_ids"]) for e in examples)
    stats["completion_tokens"] = sum(sum(1 for x in e["labels"] if x != -100) for e in examples)
    return examples, stats


def _training_order(
    examples: list[dict], args: argparse.Namespace, tcfg: dict, window: int, seed: int
) -> tuple[list[int], int]:
    """Example indices in training order (window per step) and the number of optimizer steps."""
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    if args.longest_first:
        order.sort(key=lambda i: -len(examples[i]["input_ids"]))
    if args.max_samples:
        order = order[: args.max_samples]
    epochs = int(tcfg.get("epochs", 1))
    steps_per_epoch = math.ceil(len(order) / window)
    n_steps = steps_per_epoch * epochs
    max_steps = args.max_steps or tcfg.get("max_steps")
    if max_steps:
        n_steps = min(n_steps, int(max_steps))
    # Each epoch fills whole steps (a short last window is padded with the epoch's first examples),
    # so no optimizer step ever runs on an empty window.
    padded = order + order[: steps_per_epoch * window - len(order)]
    return (padded * epochs)[: n_steps * window], n_steps


def _peak_alloc_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _device_used_gb() -> float:
    """Device-wide usage: all processes + CUDA context + bnb paged/managed memory, which
    torch's allocator stats do not see."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


def _train(
    *,
    model,
    tok,
    method: str,
    examples: list[dict],
    order: list[int],
    n_steps: int,
    bs: int,
    window: int,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    tcfg: dict,
    out_dir: Path,
    no_save: bool,
) -> tuple[float, float]:
    """Run the optimizer steps, logging each to train_log.csv. Returns (final_loss, device_used_max_gb)."""
    device = next(p for p in model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    save_every = int(tcfg.get("save_steps", 0) or 0)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    final_loss = float("nan")
    dev_used_max = 0.0

    with open(out_dir / "train_log.csv", "w", newline="") as log_f:
        log = csv.writer(log_f)
        log.writerow(LOG_COLUMNS)
        for step in range(n_steps):
            ts = time.time()
            idxs = order[step * window : (step + 1) * window]
            batches = [
                collate([examples[i] for i in idxs[j : j + bs]], tok.pad_token_id)
                for j in range(0, len(idxs), bs)
            ]
            n_comp = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
            n_tok = sum(len(examples[i]["input_ids"]) for i in idxs)
            loss_acc = 0.0
            for b in batches:
                att = b["attention_mask"]
                with autocast:
                    loss_sum, _ = completion_loss_sum(
                        model,
                        b["input_ids"].to(device),
                        b["labels"].to(device),
                        att.to(device) if att is not None else None,
                    )
                # Normalise by the whole window's completion tokens -> token-level mean per step.
                (loss_sum / n_comp).backward()
                loss_acc += loss_sum.item()
            gn = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            lr = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            dt = time.time() - ts
            final_loss = loss_acc / max(n_comp, 1)
            peak = _peak_alloc_gb()
            dev_used = _device_used_gb()
            dev_used_max = max(dev_used_max, dev_used)
            log.writerow(
                [
                    step + 1,
                    f"{final_loss:.5f}",
                    f"{lr:.3e}",
                    f"{n_tok / dt:.1f}",
                    f"{dt:.2f}",
                    n_tok,
                    n_comp,
                    f"{float(gn):.4f}",
                    f"{peak:.2f}",
                    f"{dev_used:.2f}",
                ]
            )
            log_f.flush()
            print(
                f"step {step + 1}/{n_steps} loss={final_loss:.4f} lr={lr:.2e} "
                f"tok/s={n_tok / dt:.0f} dt={dt:.1f}s gn={float(gn):.3f} peak={peak:.2f}GB "
                f"dev_used={dev_used:.2f}GB",
                flush=True,
            )
            if save_every and (step + 1) % save_every == 0 and not no_save:
                save_final(method, model, tok, out_dir / f"checkpoint-{step + 1}")
    return final_loss, dev_used_max


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    method = cfg["method"]
    tcfg = _train_config(cfg, args)
    seed = int(tcfg.get("seed", 42))
    bs = int(tcfg.get("per_device_batch_size", 1))
    ga = int(tcfg.get("grad_accum", 32))
    max_len = int(tcfg.get("max_seq_len", 3584))
    train_file = args.train_file or cfg.get("train_file", "data/processed/train.jsonl")
    out_dir = Path(args.out or f"runs/{cfg.get('run_name', method)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    t_load = time.time()
    model, tok, info = build_model(method, cfg)
    model.train()
    print(
        f"[{method}] trainable={info['trainable_params']:,} total={info['total_params']:,} "
        f"({info['pct']:.6f}%) load={time.time() - t_load:.1f}s",
        flush=True,
    )

    examples, data_stats = _prepare_examples(cfg, tcfg, tok, train_file, max_len, seed)
    print(f"data: {data_stats}", flush=True)
    window = bs * ga
    order, n_steps = _training_order(examples, args, tcfg, window, seed)
    print(f"steps={n_steps} bs={bs} ga={ga} eff_batch={window}", flush=True)

    opt, opt_name = build_optimizer(method, model, tcfg)
    warmup = math.ceil(float(tcfg.get("warmup_ratio", 0.03)) * n_steps)
    sched = cosine_with_warmup(opt, warmup, n_steps)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump({**cfg, "train": tcfg, "train_file": train_file}, sort_keys=False)
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    final_loss, dev_used_max = _train(
        model=model,
        tok=tok,
        method=method,
        examples=examples,
        order=order,
        n_steps=n_steps,
        bs=bs,
        window=window,
        opt=opt,
        sched=sched,
        tcfg=tcfg,
        out_dir=out_dir,
        no_save=args.no_save,
    )
    wall = time.time() - t0

    tokens_trained = sum(len(examples[i]["input_ids"]) for i in order)
    run_info = {
        **info,
        "config": str(args.config),
        "optimizer": opt_name,
        "lr": float(tcfg["lr"]),
        "effective_batch": window,
        "max_seq_len": max_len,
        **data_stats,
        "tokens_trained": tokens_trained,
        "steps": n_steps,
        "final_loss": final_loss,
        "wall_seconds": wall,
        "sec_per_step": wall / max(n_steps, 1),
        "tokens_per_s": tokens_trained / max(wall, 1e-9),
        "peak_vram_gb": _peak_alloc_gb(),
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0,
        "device_used_max_gb": dev_used_max,
        "train_file": train_file,
    }
    if opt_name == "paged_adamw_8bit":
        # Paged optimizer state lives in CUDA managed memory, invisible to the allocator stats above.
        run_info["optimizer_state_gb"] = (
            sum(
                t.numel() * t.element_size()
                for st in opt.state.values()
                for t in st.values()
                if torch.is_tensor(t)
            )
            / 1e9
        )
    if not args.no_save:
        save_final(method, model, tok, out_dir / "final")
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(json.dumps(run_info, indent=2), flush=True)


if __name__ == "__main__":
    main()
