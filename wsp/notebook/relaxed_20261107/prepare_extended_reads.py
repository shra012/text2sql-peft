"""Broaden alias contrasts and record first-answer instruction task vectors."""
import json
from pathlib import Path
import re
import sqlite3

from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
from prepare_probes import INSTRUCTIONS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    practice = {r["id"]: r for r in map(json.loads, (ROOT / "wsp/practice.jsonl").read_text().splitlines())}
    contrast_meta = json.loads((HERE / "contrast_metadata.json").read_text())
    pids = list(dict.fromkeys(row["original_id"] for row in contrast_meta))
    texts, metadata = [], []
    for pid in pids:
        for variant in ("original", *INSTRUCTIONS):
            messages = [dict(m) for m in practice[pid]["messages"]]
            if variant != "original":
                messages[0]["content"] += " " + INSTRUCTIONS[variant]
            label = f"first_{pid}_{variant}"
            texts.append({"id": label, "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), "answer": ""})
            metadata.append({"id": label, "kind": "first", "original_id": pid, "variant": variant})
    excluded = {r["id"].split("_", 1)[1] for r in map(json.loads, (HERE.parent / "texts_pass05.jsonl").read_text().splitlines())}
    predictions = {}
    for number in (26, 14, 18, 19):
        for row in map(json.loads, (ROOT / f"wsp/passes/pass_{number:02d}/edited.jsonl").read_text().splitlines()):
            predictions.setdefault(row["id"], row)
    catalog = {}
    count = 0
    for pid, prediction in predictions.items():
        if pid in excluded:
            continue
        raw = prediction["raw"]
        from_match = re.search(r"\bFROM\b", raw, re.I)
        if not from_match:
            continue
        prefix = raw[:from_match.end()]
        selected = re.search(r"\bT([12])\.(?:`([^`]+)`|\"([^\"]+)\"|(\w+))", prefix)
        if not selected:
            continue
        column = next(value for value in selected.groups()[1:] if value is not None).casefold()
        db = practice[pid]["db_path"]
        if db not in catalog:
            connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            tables = {}
            try:
                for table, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                    quoted = table.replace('"', '""')
                    tables[table] = {r[1].casefold() for r in connection.execute(f'PRAGMA table_info("{quoted}")')}
            finally:
                connection.close()
            catalog[db] = tables
        owners = [table for table, columns in catalog[db].items() if column in columns and re.fullmatch(r"\w+", table)]
        if len(owners) != 1:
            continue
        owner = owners[0]
        alternatives = [t for t in catalog[db] if t != owner and re.fullmatch(r"\w+", t)]
        if not alternatives:
            continue
        owner_token = tokenizer(" " + owner, add_special_tokens=False)["input_ids"][0]
        alternatives = [t for t in alternatives if tokenizer(" " + t, add_special_tokens=False)["input_ids"][0] != owner_token]
        if not alternatives:
            continue
        other = alternatives[0]
        original_digit = int(selected.group(1))
        swapped = re.sub(r"\bT([12])\b", lambda m: "T" + str(3 - int(m.group(1))), prefix)
        prompt = tokenizer.apply_chat_template(practice[pid]["messages"], tokenize=False, add_generation_prompt=True)
        for digit in (1, 2):
            label = f"binding_{pid}_{digit}"
            texts.append({"id": label, "prompt": prompt + (prefix if digit == original_digit else swapped), "answer": ""})
            metadata.append({"id": label, "kind": "binding", "original_id": pid, "digit": digit,
                             "owner": owner, "other": other})
        count += 1
        if len(texts) + 2 > 1024:
            break
    (HERE / "extended_reads.jsonl").write_text("".join(json.dumps(row) + "\n" for row in texts))
    (HERE / "extended_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Prepared", len(texts), "read texts:", len(pids)*5, "first-answer conditions and", count, "alias pairs.")


if __name__ == "__main__":
    main()
