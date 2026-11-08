"""Generate SQL with vLLM (greedy) and score BIRD execution accuracy.

    python -m t2s.evaluate --model Qwen/Qwen2.5-Coder-1.5B-Instruct --name base \
        [--split dev] [--limit N] [--gpu-mem 0.55] [--quantization bitsandbytes] \
        [--lora <adapter dir>]

Writes results/<name>/preds.jsonl and results/<name>/metrics.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from t2s.bird_eval import evaluate_batch_detailed
from t2s.data import DEFAULT_MODEL, OUT_DIR, REPO
from t2s.prompt import extract_sql

RESULTS_DIR = REPO / "results"
DIFFICULTIES = ("simple", "moderate", "challenging")


def load_split(split: str, limit: int | None) -> list[dict]:
    path = OUT_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m t2s.data --splits {split}` first")
    rows = [json.loads(line) for line in path.open()]
    return rows if limit is None else rows[:limit]


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model)
    except Exception:  # noqa: BLE001 - e.g. a merged checkpoint dir without tokenizer files
        print(f"tokenizer not found at {model}; falling back to {DEFAULT_MODEL}")
        return AutoTokenizer.from_pretrained(DEFAULT_MODEL)


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def generate(args: argparse.Namespace, rows: list[dict]) -> list[str]:
    """Greedy completions for each row's chat prompt."""
    from vllm import LLM, SamplingParams

    tok = _load_tokenizer(args.model)
    prompts = [
        tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows
    ]
    llm_kwargs = dict(
        model=args.model,
        tokenizer=args.model if tok.name_or_path == args.model else DEFAULT_MODEL,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        seed=0,
        enforce_eager=args.enforce_eager,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        cfg = json.loads((Path(args.lora) / "adapter_config.json").read_text())
        llm_kwargs.update(enable_lora=True, max_lora_rank=max(8, int(cfg.get("r", 16))), max_loras=1)
        lora_req = LoRARequest("adapter", 1, str(Path(args.lora).resolve()))

    llm = LLM(**llm_kwargs)
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens), lora_request=lora_req
    )
    texts = [o.outputs[0].text for o in outs]
    del llm
    _free_gpu()  # before the CPU-only execution phase
    return texts


def write_preds(path: Path, rows: list[dict], preds: list[str], texts: list[str], res: list[tuple]) -> None:
    with path.open("w") as f:
        for r, p, t, (ok, err) in zip(rows, preds, texts, res, strict=True):
            rec = {
                "question_id": r["question_id"],
                "db_id": r["db_id"],
                "difficulty": r["difficulty"],
                "pred_sql": p,
                "gold_sql": r["gold_sql"],
                "correct": ok,
                "error": err,
                "raw": t,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _pct(hits: list[bool]) -> float:
    return round(100 * sum(hits) / len(hits), 2)


def score(args: argparse.Namespace, rows: list[dict], res: list[tuple]) -> dict:
    """metrics.json contents, without the timing fields."""
    correct = [ok for ok, _ in res]
    metrics = {
        "name": args.name,
        "model": args.model,
        "lora": args.lora,
        "quantization": args.quantization,
        "split": args.split,
        "n": len(rows),
        "ex": _pct(correct) if rows else 0.0,
    }
    counts = {}
    for d in DIFFICULTIES:
        sel = [ok for r, ok in zip(rows, correct, strict=True) if r["difficulty"] == d]
        metrics[f"ex_{d}"] = _pct(sel) if sel else None
        counts[d] = {"n": len(sel), "correct": int(sum(sel))}
    metrics["counts"] = counts
    metrics["n_correct"] = int(sum(correct))
    metrics["n_pred_errors"] = sum(1 for ok, e in res if not ok and e.startswith("pred:"))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF name or local path")
    ap.add_argument("--name", required=True, help="run name -> results/<name>/")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--quantization", default=None, help="e.g. bitsandbytes")
    ap.add_argument("--lora", default=None, help="PEFT LoRA adapter dir")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    rows = load_split(args.split, args.limit)
    texts = generate(args, rows)
    gen_s = time.time() - t0

    preds = [extract_sql(t) for t in texts]
    items = [
        {"db_path": r["db_path"], "pred_sql": p, "gold_sql": r["gold_sql"]}
        for r, p in zip(rows, preds, strict=True)
    ]
    t1 = time.time()
    res = evaluate_batch_detailed(items, num_workers=args.num_workers)
    exec_s = time.time() - t1

    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_preds(out_dir / "preds.jsonl", rows, preds, texts, res)
    metrics = score(args, rows, res)
    metrics["gen_seconds"] = round(gen_s, 1)
    metrics["exec_seconds"] = round(exec_s, 1)
    metrics["eval_seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(
        f"[{args.name}] EX={metrics['ex']:.2f}% (n={len(rows)}) simple={metrics['ex_simple']} "
        f"moderate={metrics['ex_moderate']} challenging={metrics['ex_challenging']} "
        f"pred_errors={metrics['n_pred_errors']} time={metrics['eval_seconds']}s -> {out_dir}"
    )


if __name__ == "__main__":
    main()
