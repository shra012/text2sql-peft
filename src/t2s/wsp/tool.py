"""Forward-pass tool for weight-space programming (the only way the editor may run the model).

Every successful `gen` or `read` uses one pass of the budget (default 25, env WSP_BUDGET); failed calls
are free. Practice data: wsp/practice.jsonl (BIRD *train* questions only; the dev set is never used
here). Outputs land in wsp/passes/pass_NN/.

Commands (python -m t2s.wsp.tool <command>):
  prepare                       build wsp/practice.jsonl from BIRD train (prompts <= 2048 tokens)
  extend-practice               append longer train prompts (--max-tokens, default 3072) as new ids
  status                        budget, saved edits and the last five passes
  snapshot NAME                 save the current edits under a name (free)
  restore NAME                  make a snapshot the current edits again (free)
  gen                           greedy generations, graded by execution accuracy (one pass)
      --which original|edited|both  --vs original|SNAPSHOT  --protocol A|B
      --sample 256 --seed 0 | --ids p00012,p00345 | --ids-file f.txt
  read                          activations / log-probs of fixed texts (one pass)
      --model original|edited  --capture logprobs,resid:L,mlp_act:L,attn:L  --positions answer|last|all
      --sample 64 --seed 0 | --ids ... | --ids-file ... | --texts-file my_texts.jsonl
  submit [--note "why"]         hand the current edits to the harness owner (no passes after this)

Examples:
  python -m t2s.wsp.tool gen --which both --sample 256
  python -m t2s.wsp.tool gen --which both --vs F4 --sample 256   # current edits vs a snapshot, paired
  python -m t2s.wsp.tool gen --which both --vs F4 --protocol A   # pre-registered held-out test
  python -m t2s.wsp.tool read --model original --sample 64 --capture logprobs,resid:15,mlp_act:15,attn:15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

from t2s.wsp.common import (
    EDITS,
    ROOT,
    WSP,
    append_edit_log,
    base_dir,
    build_merged,
    edits_fingerprint,
    list_edits,
    timestamp,
)

BUDGET = int(os.environ.get("WSP_BUDGET", "25"))
MAX_GEN, MAX_READ, MAX_READ_ALL = 512, 1024, 64  # items per pass (read --positions all: MAX_READ_ALL)

TRAIN = ROOT / "data/processed/train.jsonl"
LEDGER = WSP / "ledger.json"
PRACTICE = WSP / "practice.jsonl"
MERGED = WSP / "_merged"
ORIGINAL_CACHE = WSP / "cache_original_gen.jsonl"
SNAPS = WSP / "snapshots"
PROTO = WSP / "protocol"
PROTO_RESULTS = PROTO / "results.jsonl"

PROTOCOL_REF = "F4"  # snapshot every protocol candidate is compared with
PROTOCOL_MAX_CANDIDATES = 4
PROTOCOL_GATE_A = 8  # net (fixed - broke) on cohort A needed before cohort B
CAPTURE_KINDS = ("resid", "mlp_act", "attn")


# ------------------------------------------------------------------ files
def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _write_jsonl(path: str | Path, records: Iterable[dict], mode: str = "w") -> None:
    with open(path, mode) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_ids(path: str | Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ------------------------------------------------------------------ ledger
def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"budget": BUDGET, "used": 0, "passes": [], "submitted": None}


def save_ledger(led: dict) -> None:
    LEDGER.write_text(json.dumps(led, indent=1))


def guard_budget(led: dict) -> None:
    if led.get("submitted"):
        sys.exit("already submitted; no more passes")
    if led["used"] >= led["budget"]:
        sys.exit(
            f"budget exhausted ({led['used']}/{led['budget']} passes). "
            "Submit with: python -m t2s.wsp.tool submit"
        )


def new_pass_dir(led: dict) -> Path:
    out_dir = WSP / "passes" / f"pass_{led['used'] + 1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def charge(led: dict, kind: str, args: argparse.Namespace, out_dir: Path, summary: dict) -> None:
    """Record a successful pass in the ledger."""
    led["used"] += 1
    led["passes"].append(
        {
            "n": led["used"],
            "kind": kind,
            "t": timestamp(),
            "args": {k: v for k, v in vars(args).items() if k != "func"},
            "edits": list_edits(),
            "edits_fingerprint": edits_fingerprint(),
            "out": str(out_dir),
            "summary": summary,
        }
    )
    save_ledger(led)
    print(f"[budget] used {led['used']}/{led['budget']} passes")


# ------------------------------------------------------------------ practice data
def load_practice() -> list[dict]:
    return _read_jsonl(PRACTICE)


def cohort_ids() -> set[str]:
    """Ids of the held-out protocol cohorts (usable only through --protocol)."""
    ids = set()
    for name in ("A", "B"):
        path = PROTO / f"cohort_{name}.txt"
        if path.exists():
            ids.update(_read_ids(path))
    return ids


def _requested_ids(args: argparse.Namespace, by_id: dict, held_out: set[str]) -> list[str]:
    if getattr(args, "protocol", None):
        return _read_ids(PROTO / f"cohort_{args.protocol}.txt")
    if getattr(args, "ids", None):
        return [i.strip() for i in args.ids.split(",") if i.strip()]
    if getattr(args, "ids_file", None):
        return _read_ids(args.ids_file)
    # Counterfactual (probe_only) prompts are available only through explicit ids; they must
    # never enter ordinary random practice/evaluation cohorts.
    ids = sorted(i for i, row in by_id.items() if not row.get("probe_only", False) and i not in held_out)
    random.Random(args.seed).shuffle(ids)
    return ids[: args.sample]


def pick(rows: list[dict], args: argparse.Namespace, cap: int) -> list[dict]:
    """Rows selected by --protocol, --ids, --ids-file or a seeded random --sample (at most `cap`)."""
    by_id = {r["id"]: r for r in rows}
    held_out = cohort_ids()
    ids = _requested_ids(args, by_id, held_out)
    if not getattr(args, "protocol", None) and held_out.intersection(ids):
        sys.exit("those ids belong to the held-out protocol cohorts; use --protocol A|B (gen only)")
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f"unknown practice ids: {missing[:5]}")
    if len(ids) > cap:
        sys.exit(f"at most {cap} items per pass (got {len(ids)})")
    return [by_id[i] for i in ids]


def _train_rows(max_tokens: int) -> list[dict]:
    """BIRD train rows whose prompt was not truncated and fits in max_tokens."""
    return [
        r for r in _read_jsonl(TRAIN) if r["fit_level"] != "truncated" and r["prompt_tokens"] <= max_tokens
    ]


def _practice_record(pid: str, r: dict) -> dict:
    keys = ("db_id", "db_path", "question", "evidence", "gold_sql", "messages", "completion", "prompt_tokens")
    return {"id": pid, **{k: r[k] for k in keys}}


def prepare(args: argparse.Namespace) -> None:
    if PRACTICE.exists() and not args.force:
        sys.exit(f"{PRACTICE} exists; rebuilding renumbers ids (breaks passes and cohorts). Use --force.")
    rows = _train_rows(2048)
    WSP.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PRACTICE, (_practice_record(f"p{i:05d}", r) for i, r in enumerate(rows)))
    print(f"wrote {len(rows)} practice questions (BIRD train, prompts <= 2048 tokens) -> {PRACTICE}")


def extend_practice(args: argparse.Namespace) -> None:
    """Append train questions with longer prompts (up to --max-tokens) as new ids.

    Existing ids keep their meaning.
    """
    existing = load_practice()
    have = {(r["db_id"], r["question"], r["gold_sql"]) for r in existing}
    n0 = len(existing)
    new = [r for r in _train_rows(args.max_tokens) if (r["db_id"], r["question"], r["gold_sql"]) not in have]
    _write_jsonl(PRACTICE, (_practice_record(f"p{n0 + i:05d}", r) for i, r in enumerate(new)), mode="a")
    print(
        f"appended {len(new)} practice questions (prompts up to {args.max_tokens} tokens): "
        f"ids p{n0:05d}..p{n0 + len(new) - 1:05d}"
    )


# ------------------------------------------------------------------ snapshots
def _snapshot_names() -> list[str]:
    return sorted(p.name for p in SNAPS.iterdir()) if SNAPS.exists() else []


def snapshot(args: argparse.Namespace) -> None:
    if not list_edits():
        sys.exit("no edits to snapshot")
    snap = SNAPS / args.name
    if snap.exists():
        sys.exit(f"snapshot {args.name} exists; pick a new name")
    shutil.copytree(EDITS, snap)
    print(f"snapshot {args.name}: {list_edits(snap)} (fingerprint {edits_fingerprint(snap)})")


def restore(args: argparse.Namespace) -> None:
    snap = SNAPS / args.name
    if not snap.exists():
        sys.exit(f"no snapshot {args.name}; have {_snapshot_names()}")
    shutil.rmtree(EDITS, ignore_errors=True)
    shutil.copytree(snap, EDITS)
    append_edit_log({"action": "restore", "snapshot": args.name})
    print(f"current edits <- snapshot {args.name}: {list_edits()}")


# ------------------------------------------------------------------ gen
def _vllm_generate(model_dir: str, rows: list[dict]) -> list[str]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [
        tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows
    ]
    llm = LLM(model=model_dir, gpu_memory_utilization=0.6, max_model_len=4096, seed=0, enforce_eager=False)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
    texts = [o.outputs[0].text for o in outs]
    del llm
    return texts


def _grade(rows: list[dict], texts: list[str]) -> list[dict]:
    """One record per row: execution-match against the gold SQL plus format statistics."""
    from t2s.bird_eval import evaluate_batch_detailed
    from t2s.prompt import extract_sql

    preds = [extract_sql(t) for t in texts]
    items = [
        {"db_path": r["db_path"], "pred_sql": p, "gold_sql": r["gold_sql"]}
        for r, p in zip(rows, preds, strict=False)
    ]
    results = evaluate_batch_detailed(items, num_workers=8)
    return [
        {
            "id": r["id"],
            "correct": bool(ok),
            "error": err or "",
            "pred_sql": p,
            "raw": t,
            "chars": len(t),
            "sql_first": t.lstrip().startswith("```sql"),
        }
        for r, t, p, (ok, err) in zip(rows, texts, preds, results, strict=False)
    ]


def _stats(recs: list[dict]) -> dict:
    n = len(recs)
    if not n:
        return {}
    errs = [r["error"] for r in recs if r["error"].startswith("pred:")]
    return {
        "n": n,
        "acc": round(sum(r["correct"] for r in recs) / n, 4),
        "pred_errors": len(errs),
        "no_such_column": sum("no such column" in e for e in errs),
        "no_such_table": sum("no such table" in e for e in errs),
        "mean_chars": round(statistics.mean(r["chars"] for r in recs), 1),
        "sql_first": round(sum(r["sql_first"] for r in recs) / n, 4),
    }


def _passed_gate_a(results: list[dict], fp: str) -> bool:
    return any(r["fingerprint"] == fp and r["cohort"] == "A" and r["net"] >= PROTOCOL_GATE_A for r in results)


def protocol_check(args: argparse.Namespace) -> str:
    """Pre-registered gate: <=4 candidates, compared with snapshot F4; B only after passing A.

    Returns the fingerprint of the current edits (the candidate).
    """
    if args.which != "both" or args.vs != PROTOCOL_REF:
        sys.exit(f"protocol tests must be: gen --which both --vs {PROTOCOL_REF} --protocol A|B")
    fp = edits_fingerprint()
    results = _read_jsonl(PROTO_RESULTS) if PROTO_RESULTS.exists() else []
    candidates = {r["fingerprint"] for r in results}
    if fp in {r["fingerprint"] for r in results if r["cohort"] == args.protocol}:
        sys.exit("this candidate was already tested on that cohort")
    if args.protocol == "A" and fp not in candidates and len(candidates) >= PROTOCOL_MAX_CANDIDATES:
        sys.exit(f"protocol allows at most {PROTOCOL_MAX_CANDIDATES} candidates")
    if args.protocol == "B" and not _passed_gate_a(results, fp):
        sys.exit(
            "cohort B is only for candidates that passed cohort A "
            f"(net >= +{PROTOCOL_GATE_A} vs {PROTOCOL_REF})"
        )
    return fp


def _reference_outputs(vs: str, rows: list[dict]) -> list[dict]:
    """Graded outputs of the reference model (base, or snapshot `vs`), cached per reference weights."""
    if vs == "original":
        cache_file, model_dir = ORIGINAL_CACHE, str(base_dir())
    else:
        snap = SNAPS / vs
        if not snap.exists():
            sys.exit(f"no snapshot {vs}")
        cache_file, model_dir = WSP / f"cache_snapshot_{vs}_{edits_fingerprint(snap)}.jsonl", None
    cache = {rec["id"]: rec for rec in _read_jsonl(cache_file)} if cache_file.exists() else {}
    todo = [r for r in rows if r["id"] not in cache]
    if todo:
        if model_dir is None:
            model_dir = str(build_merged(WSP / f"_merged_snapshot_{vs}", snap))
        graded = _grade(todo, _vllm_generate(model_dir, todo))
        _write_jsonl(cache_file, graded, mode="a")
        cache.update((rec["id"], rec) for rec in graded)
    return [cache[r["id"]] for r in rows]


def _record_protocol_result(
    args: argparse.Namespace, fp: str, pass_n: int, ref_name: str, summary: dict
) -> None:
    paired = summary["paired"]
    net = paired["fixed"] - paired["broke"]
    record = {
        "t": timestamp(),
        "cohort": args.protocol,
        "fingerprint": fp,
        "edits": list_edits(),
        "pass": pass_n,
        **paired,
        "net": net,
        "acc_ref": summary[ref_name]["acc"],
        "acc_edited": summary["edited"]["acc"],
    }
    _write_jsonl(PROTO_RESULTS, [record], mode="a")
    gate = " -> passes gate A, eligible for B" if args.protocol == "A" and net >= PROTOCOL_GATE_A else ""
    print(f"[protocol] cohort {args.protocol}: net {net:+d} vs {PROTOCOL_REF}{gate}")


def gen(args: argparse.Namespace) -> None:
    led = load_ledger()
    guard_budget(led)
    fp = protocol_check(args) if args.protocol else None
    rows = pick(load_practice(), args, MAX_GEN)
    out_dir = new_pass_dir(led)
    ref_name = "original" if args.vs == "original" else f"snapshot_{args.vs}"
    results = {}
    if args.which in ("original", "both"):
        results[ref_name] = _reference_outputs(args.vs, rows)
    if args.which in ("edited", "both"):
        if not list_edits():
            sys.exit("no edits saved yet; use --which original, or save_edit() first")
        build_merged(MERGED)
        results["edited"] = _grade(rows, _vllm_generate(str(MERGED), rows))

    summary = {k: _stats(v) for k, v in results.items()}
    if len(results) == 2:
        ref = {r["id"]: r["correct"] for r in results[ref_name]}
        edited = {r["id"]: r["correct"] for r in results["edited"]}
        summary["paired"] = {
            "fixed": sum(edited[i] and not ref[i] for i in ref),
            "broke": sum(ref[i] and not edited[i] for i in ref),
        }
    for name, recs in results.items():
        _write_jsonl(out_dir / f"{name}.jsonl", recs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if fp:
        _record_protocol_result(args, fp, led["used"] + 1, ref_name, summary)
    charge(led, "gen", args, out_dir, summary)
    print(json.dumps(summary, indent=1))
    print(f"per-question outputs: {out_dir}/{{{ref_name},edited}}.jsonl")


# ------------------------------------------------------------------ read
def _parse_captures(spec: str) -> list[str]:
    caps = [c.strip() for c in spec.split(",") if c.strip()]
    for c in caps:
        kind, _, layer = c.partition(":")
        if c != "logprobs" and not (kind in CAPTURE_KINDS and layer.isdigit()):
            sys.exit(f"unknown capture {c}; use logprobs, resid:L, mlp_act:L, attn:L")
    if len(set(caps)) != len(caps):
        sys.exit("each capture may appear only once")
    return caps


def _layer(capture: str) -> int:
    return int(capture.split(":")[1])


def _text_item(i: int, it: dict) -> dict:
    """A --texts-file line: {"prompt", "answer"} (prompt optional) or {"text"}; "id" optional."""
    if "answer" in it:
        prompt, answer = it.get("prompt", ""), it["answer"]
    else:
        prompt, answer = "", it["text"]
    return {"id": it.get("id", f"t{i}"), "prompt": prompt, "answer": answer}


def _read_items(args: argparse.Namespace, cap: int) -> tuple[list[dict], bool]:
    """Items to read and whether they are chat prompts (practice rows) rather than raw texts."""
    if args.texts_file:
        texts = _read_jsonl(args.texts_file)
        if len(texts) > cap:
            sys.exit(f"at most {cap} texts per pass")
        return [_text_item(i, it) for i, it in enumerate(texts)], False
    rows = pick(load_practice(), args, cap)
    return [{"id": r["id"], "messages": r["messages"], "answer": r["completion"]} for r in rows], True


def _token_ids(tok, item: dict, chat: bool) -> tuple[list[int], list[int]]:
    """(prompt ids, answer ids); a chat answer ends with <|im_end|> as in training."""
    if chat:
        prompt_ids = tok.apply_chat_template(item["messages"], add_generation_prompt=True, tokenize=True)
        answer_ids = tok(item["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    else:
        prompt_ids = tok(item["prompt"], add_special_tokens=False)["input_ids"] if item["prompt"] else []
        answer_ids = tok(item["answer"], add_special_tokens=False)["input_ids"]
    return prompt_ids, answer_ids


def _kept_positions(mode: str, n_prompt: int, has_answer: bool, n_total: int) -> list[int]:
    """Query positions to keep; "answer" = the positions that predict an answer token."""
    if mode == "answer":
        return list(range(max(n_prompt, 1) - 1, n_total - 1)) if has_answer else [n_total - 1]
    if mode == "last":
        return [n_total - 1]
    return list(range(n_total))


def _capture(model, tok, items: list[dict], chat: bool, caps: list[str], mode: str) -> dict:
    import torch

    layers = model.model.layers
    want_resid = any(c.startswith("resid:") for c in caps)
    want_attn = any(c.startswith("attn:") for c in caps)
    store = {"ids": [], "tokens": [], "positions": [], **{c: [] for c in caps}}
    mlp_inputs = {}

    def save_mlp_input(layer):
        def hook(_module, inputs):
            mlp_inputs[layer] = inputs[0].detach()

        return hook

    hooks = [
        layers[_layer(c)].mlp.down_proj.register_forward_pre_hook(save_mlp_input(_layer(c)))
        for c in caps
        if c.startswith("mlp_act:")
    ]
    with torch.no_grad():
        for item in items:
            prompt_ids, answer_ids = _token_ids(tok, item, chat)
            ids = torch.tensor([prompt_ids + answer_ids]).cuda()
            n_total = ids.shape[1]
            pos = _kept_positions(mode, len(prompt_ids), bool(answer_ids), n_total)
            out = model(ids, output_hidden_states=want_resid, output_attentions=want_attn)
            store["ids"].append(item["id"])
            store["tokens"].append(ids[0].cpu())
            store["positions"].append(torch.tensor(pos))
            for c in caps:
                if c == "logprobs":  # log-prob of the actual next token at each kept position
                    logp = torch.log_softmax(out.logits[0, pos].float(), -1)
                    nxt = ids[0, [min(p + 1, n_total - 1) for p in pos]]
                    store[c].append(logp.gather(-1, nxt[:, None])[:, 0].cpu())
                elif c.startswith("resid:"):  # output of decoder layer L
                    store[c].append(out.hidden_states[_layer(c) + 1][0, pos].float().cpu().half())
                elif c.startswith("mlp_act:"):  # input to down_proj (after gate*up)
                    store[c].append(mlp_inputs[_layer(c)][0, pos].float().cpu().half())
                elif c.startswith("attn:"):  # (heads, kept query positions, keys)
                    store[c].append(out.attentions[_layer(c)][0][:, pos].float().cpu().half())
    for h in hooks:
        h.remove()
    return store


def read(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    led = load_ledger()
    guard_budget(led)
    caps = _parse_captures(args.capture)
    items, chat = _read_items(args, MAX_READ_ALL if args.positions == "all" else MAX_READ)
    if args.model == "edited" and not list_edits():
        sys.exit("no edits saved yet")
    model_dir = str(build_merged(MERGED)) if args.model == "edited" else str(base_dir())
    tok = AutoTokenizer.from_pretrained(model_dir)
    attn_impl = "eager" if any(c.startswith("attn:") for c in caps) else "sdpa"
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation=attn_impl)
        .cuda()
        .eval()
    )
    store = _capture(model, tok, items, chat, caps, args.positions)

    out_dir = new_pass_dir(led)
    torch.save(store, out_dir / "read.pt")
    summary = {"model": args.model, "n_texts": len(items), "captures": caps, "positions": args.positions}
    if "logprobs" in caps:
        summary["mean_answer_logprob"] = round(float(torch.cat(store["logprobs"]).mean()), 4)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    charge(led, "read", args, out_dir, summary)
    print(json.dumps(summary, indent=1))
    print(f"tensors: torch.load('{out_dir}/read.pt')  keys: ids, tokens, positions, {', '.join(caps)}")


# ------------------------------------------------------------------ status / submit
def status(_args: argparse.Namespace) -> None:
    led = load_ledger()
    edits = list_edits()
    print(f"budget: {led['used']}/{led['budget']} passes used; submitted: {led.get('submitted')}")
    print(f"saved edits ({len(edits)}): {edits}")
    for p in led["passes"][-5:]:
        print(f"  pass {p['n']} {p['kind']} {p['t']}: {json.dumps(p['summary'])[:300]}")


def submit(args: argparse.Namespace) -> None:
    led = load_ledger()
    edits = list_edits()
    if not edits:
        sys.exit("nothing to submit: no edits saved")
    fp = edits_fingerprint()
    led["submitted"] = {"t": timestamp(), "note": args.note, "edits": edits, "edits_fingerprint": fp}
    save_ledger(led)
    print(f"submitted {len(edits)} edited tensors (fingerprint {fp}). The harness owner evaluates it.")


# ------------------------------------------------------------------ CLI
def _add_selection_args(p: argparse.ArgumentParser, sample: int) -> None:
    p.add_argument("--sample", type=int, default=sample, help="random practice questions (seeded)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ids", help="comma-separated practice ids")
    p.add_argument("--ids-file", help="file with one practice id per line")


def build_parser() -> argparse.ArgumentParser:
    # Argument order matters: vars(args) is stored in ledger.json in this order.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)

    s = sub.add_parser("prepare")
    s.add_argument("--force", action="store_true", help="overwrite an existing practice set")
    s.set_defaults(func=prepare)
    sub.add_parser("status").set_defaults(func=status)

    p = sub.add_parser("extend-practice")
    p.add_argument("--max-tokens", type=int, default=3072)
    p.set_defaults(func=extend_practice)

    for name, func in (("snapshot", snapshot), ("restore", restore)):
        p = sub.add_parser(name)
        p.add_argument("name")
        p.set_defaults(func=func)

    p = sub.add_parser("gen")
    p.add_argument("--which", choices=["original", "edited", "both"], default="both")
    p.add_argument(
        "--vs", default="original", help="reference for 'original'/'both': original or a snapshot name"
    )
    p.add_argument(
        "--protocol", choices=["A", "B"], help="pre-registered held-out test vs F4 (see wsp/protocol)"
    )
    _add_selection_args(p, sample=256)
    p.set_defaults(func=gen)

    p = sub.add_parser("read")
    p.add_argument("--model", choices=["original", "edited"], default="original")
    p.add_argument("--capture", default="logprobs")
    p.add_argument("--positions", choices=["answer", "last", "all"], default="answer")
    _add_selection_args(p, sample=64)
    p.add_argument("--texts-file", help="jsonl of {prompt, answer} or {text} instead of practice questions")
    p.set_defaults(func=read)

    p = sub.add_parser("submit")
    p.add_argument("--note", default="")
    p.set_defaults(func=submit)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
