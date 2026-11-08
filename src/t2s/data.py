"""Build data/processed/{dev,train,train_heldout200}.jsonl from BIRD.

    python -m t2s.data [--splits dev train] [--max-tokens 3072] [--tokenizer NAME]

Each row: question_id, db_id, db_path (absolute), question, evidence, gold_sql, difficulty, messages,
completion, prompt_tokens, fit_level (how the prompt was fitted to the token budget).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from t2s.prompt import fit_prompt_info, format_completion

REPO = Path(__file__).resolve().parents[2]
BIRD_DIR = REPO / "data" / "bird"
OUT_DIR = REPO / "data" / "processed"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
HELDOUT_N = 200
HELDOUT_SEED = 42


def _find_one(root: Path, name: str, want_dir: bool = False) -> Path:
    """Shallowest match for `name` under `root`, ignoring macOS archive junk."""
    hits = [
        p for p in root.rglob(name) if "__MACOSX" not in p.parts and (p.is_dir() if want_dir else p.is_file())
    ]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return min(hits, key=lambda p: len(p.parts))


def split_paths(split: str) -> tuple[Path, Path]:
    """(questions json, databases dir) for 'dev' or 'train'."""
    root = BIRD_DIR / split
    return _find_one(root, f"{split}.json"), _find_one(root, f"{split}_databases", want_dir=True)


def load_raw(split: str) -> list[dict]:
    """Raw BIRD rows with absolute db_path, gold_sql and difficulty filled in."""
    qpath, dbdir = split_paths(split)
    return [
        {
            "question_id": r.get("question_id", i),
            "db_id": r["db_id"],
            "db_path": str((dbdir / r["db_id"] / f"{r['db_id']}.sqlite").resolve()),
            "question": r["question"],
            "evidence": r.get("evidence", "") or "",
            "gold_sql": r["SQL"].strip(),
            "difficulty": r.get("difficulty", "unknown") or "unknown",
        }
        for i, r in enumerate(json.loads(qpath.read_text()))
    ]


def build_rows(rows: list[dict], tokenizer, max_tokens: int) -> list[dict]:
    """Add the fitted chat prompt and target completion to each row; drop rows whose DB is missing."""
    out = []
    missing = set()
    for k, r in enumerate(rows):
        if not Path(r["db_path"]).exists():
            missing.add(r["db_id"])
            continue
        msgs, level, n_tok = fit_prompt_info(
            tokenizer, r["db_path"], r["question"], r["evidence"], max_tokens
        )
        out.append(
            {
                **r,
                "messages": msgs,
                "completion": format_completion(r["gold_sql"]),
                "prompt_tokens": n_tok,
                "fit_level": level,
            }
        )
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(rows)}", flush=True)
    if missing:
        print(f"  WARNING: skipped rows for missing DBs: {sorted(missing)}")
    return out


def split_heldout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(train rows, held-out rows): a fixed random sample of HELDOUT_N rows is held out."""
    held_idx = set(random.Random(HELDOUT_SEED).sample(range(len(rows)), HELDOUT_N))
    train = [r for i, r in enumerate(rows) if i not in held_idx]
    held = [r for i, r in enumerate(rows) if i in held_idx]
    return train, held


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats(name: str, rows: list[dict]) -> None:
    toks = sorted(r["prompt_tokens"] for r in rows)
    if not toks:
        print(f"{name}: 0 rows")
        return
    p95 = toks[min(len(toks) - 1, int(round(0.95 * (len(toks) - 1))))]
    levels = Counter(r["fit_level"] for r in rows)
    diff = Counter(r["difficulty"] for r in rows)
    print(
        f"{name}: n={len(rows)} dbs={len({r['db_id'] for r in rows})} "
        f"prompt_tokens mean={statistics.mean(toks):.0f} median={statistics.median(toks):.0f} "
        f"p95={p95} max={toks[-1]} | fit_levels={dict(levels)} | difficulty={dict(diff)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=["dev", "train"], choices=["dev", "train"])
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--tokenizer", default=DEFAULT_MODEL)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    for split in args.splits:
        print(f"[{split}] building ...", flush=True)
        rows = build_rows(load_raw(split), tok, args.max_tokens)
        if split == "train":
            rows, held = split_heldout(rows)
            write_jsonl(OUT_DIR / "train_heldout200.jsonl", held)
            stats("train_heldout200", held)
        write_jsonl(OUT_DIR / f"{split}.jsonl", rows)
        stats(split, rows)


if __name__ == "__main__":
    main()
