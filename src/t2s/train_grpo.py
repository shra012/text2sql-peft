"""GRPO for TinyLoRA on BIRD train with an execution-match reward, no KL and colocated vLLM rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from t2s.bird_eval import evaluate_batch
from t2s.prompt import extract_sql
from t2s.tinylora import apply_tinylora, merged_weights, save_tinylora


def load_rows(path: str, max_prompt_tokens: int, limit: int | None, seed: int) -> Dataset:
    """Rows with prompt_tokens <= max_prompt_tokens, shuffled with `seed`, first `limit` kept."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["prompt_tokens"] <= max_prompt_tokens:
                rows.append({"prompt": r["messages"], "db_path": r["db_path"], "gold_sql": r["gold_sql"]})
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    return Dataset.from_list(rows)


def make_reward(num_workers: int, timeout: float):
    """Binary reward: 1.0 if the extracted SQL's execution result matches the gold SQL's."""

    def execution_reward(prompts, completions, db_path, gold_sql, **kwargs):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        items = [
            {"db_path": d, "pred_sql": extract_sql(t), "gold_sql": g, "timeout": timeout}
            for t, d, g in zip(texts, db_path, gold_sql, strict=False)
        ]
        return [1.0 if ok else 0.0 for ok in evaluate_batch(items, num_workers=num_workers)]

    return execution_reward


class TinyLoRAGRPOTrainer(GRPOTrainer):
    """Pushes the merged adapted weights W' = W + US(sum v_i P_i)V^T into the colocated vLLM engine."""

    def _move_model_to_vllm(self):
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        with torch.no_grad():
            weights = merged_weights(self.accelerator.unwrap_model(self.model))
            llm_model.load_weights(list(weights.items()))
        self.llm.reset_prefix_cache()


class CsvLogger(TrainerCallback):
    """Appends the numeric entries of every log (plus v, when it is a single small vector) to
    train_log.csv, and saves a TinyLoRA checkpoint alongside each Trainer save."""

    def __init__(self, out: Path, model):
        self.path = out / "train_log.csv"
        self.path.unlink(missing_ok=True)
        self.fields: list[str] | None = None
        self.model = model
        self.out = out

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if "train_runtime" in numeric:
            (self.out / "train_summary.json").write_text(json.dumps(numeric, indent=2))
            return
        row = {"step": state.global_step, "wall": round(time.time(), 1), **numeric}
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable) == 1 and trainable[0].numel() <= 64:
            row["v"] = json.dumps([round(x, 6) for x in trainable[0].detach().float().cpu().tolist()])
        new_file = self.fields is None
        self.fields = self.fields or list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)

    def on_save(self, args, state, control, **kw):
        save_tinylora(self.model, self.out / f"checkpoint-{state.global_step}")


def _grpo_config(cfg: dict, out: Path, max_steps: int | None, seed: int) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(out),
        learning_rate=cfg["lr"],
        weight_decay=0.0,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=1,
        max_steps=max_steps or cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_tokens"],
        max_completion_length=cfg["max_completion_tokens"],
        temperature=cfg.get("temperature", 1.0),
        beta=cfg.get("beta", 0.0),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_mem", 0.35),
        vllm_importance_sampling_correction=True,
        # Offload vLLM weights and drop its KV cache during fwd/bwd, so nothing spills on a 16 GB GPU.
        vllm_enable_sleep_mode=cfg.get("vllm_sleep", True),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 50),
        save_only_model=True,
        report_to="none",
        seed=seed,
        log_completions=False,
    )


def _hard_exit() -> None:
    """vLLM colocate leaves a process group and engine alive whose teardown at interpreter exit
    aborts (rc 134) after everything is saved; destroy the group explicitly, then exit 0."""
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/tinylora_grpo.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--out", default=None, help="run dir (default runs/<name>)")
    ap.add_argument("--lr", type=float, default=None, help="override lr")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.lr is not None:
        cfg["lr"] = args.lr
    out = Path(args.out) if args.out else Path(f"runs/{cfg['name']}")
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map="cuda")
    model = apply_tinylora(model, u=cfg["u"], r=cfg["r"], tie=cfg["tie"], seed=cfg.get("tinylora_seed", 0))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train}")

    ds = load_rows(cfg["train_file"], cfg["max_prompt_tokens"], cfg.get("limit"), seed)
    print(f"GRPO prompts: {len(ds)}")

    trainer = TinyLoRAGRPOTrainer(
        model=model,
        reward_funcs=[make_reward(cfg.get("reward_workers", 12), cfg.get("sql_timeout", 15.0))],
        args=_grpo_config(cfg, out, args.max_steps, seed),
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.add_callback(CsvLogger(out, model))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    save_tinylora(model, out / "final")
    info = {
        "method": "tinylora_grpo",
        "trainable_params": n_train,
        "pct": 100 * n_train / sum(p.numel() for p in model.parameters()),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "wall_seconds": wall,
        "steps": trainer.state.global_step,
        "config": cfg,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    _hard_exit()


if __name__ == "__main__":
    main()
